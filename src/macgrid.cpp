#include "macgrid.h"

#include <Eigen/IterativeLinearSolvers>
#include <PolyVox/MarchingCubesSurfaceExtractor.h>
#include <PolyVox/Mesh.h>
#include <cmath>
#include <functional>
#include <igl/floor.h>
#include <iostream>

using namespace PolyVox;
using namespace Eigen;
using namespace std;

MaCGrid::MaCGrid(const double _h, const double _viscocity, const double _density)
	: h(_h), viscocity(_viscocity), density(_density), marker_particles(0, 3),
	  volData(Region(Vector3DInt32(-100, 0, -100), Vector3DInt32(100, 100, 100)))
{
	volData.setBorderValue(GridCell(Vector3i::Zero(), -1, SOLID));

	Region region = volData.getEnclosingRegion();
	int32_t z, y, x;
#pragma omp parallel for private(z)
	for (z = region.getLowerZ(); z < region.getUpperZ(); z++)
#pragma omp parallel for private(y) shared(z)
		for (y = region.getLowerY(); y < region.getUpperY(); y++)
#pragma omp parallel for private(x) shared(z, y)
			for (x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				auto &cell = volData.getVoxelRef(x, y, z);
				cell.coord << x, y, z;

				// Make an (invisible) solid floor
				if (x * x + y * y + z * z < 100)
					cell.type = SOLID;
			}
}

void MaCGrid::addParticles(const Eigen::MatrixXd &positions)
{
	int oldRows = marker_particles.rows();
	marker_particles.conservativeResize(oldRows + positions.rows(), 3);
	marker_particles.block(oldRows, 0, positions.rows(), 3) = positions;
}

void MaCGrid::simulate(const double timestep)
{
	updateGrid();
	advanceField(timestep);
	moveParticles(timestep);
}

igl::opengl::ViewerData &MaCGrid::displayVoxelMesh(igl::opengl::glfw::Viewer &viewer,
												   const int offset, CellType voxelType)
{
	CustomController controller(voxelType);
	auto mesh = extractMarchingCubesMesh(&volData, volData.getEnclosingRegion(), controller);
	auto decoded = decodeMesh(mesh);
	MatrixXd V(decoded.getNoOfVertices(), 3);
	MatrixXd colors(decoded.getNoOfVertices(), 3);
	MatrixXi T(decoded.getNoOfIndices() / 3, 3);
	for (int i = 0; i < V.rows(); i++)
	{
		auto vertex = decoded.getVertex(i).position +
					  (Vector3DFloat)volData.getEnclosingRegion().getLowerCorner();
		V.row(i) << vertex.getX(), vertex.getY(), vertex.getZ();
		colors.row(i) << V.row(i) / 2.0;
		colors.row(i).y() /= 10.0;
	}
	for (int i = 0; i < decoded.getNoOfIndices(); i++)
		T(i / 3, i % 3) = (int)decoded.getIndex(i);

	auto &viewData = viewer.data_list[offset];
	viewData.clear();
	viewData.set_mesh(V, T);
	viewData.set_face_based(true);
	viewData.set_colors(colors);
	viewData.show_lines = false;
	return viewData;
}

void MaCGrid::displayFluid(igl::opengl::glfw::Viewer &viewer, const int offset)
{
	auto &fluidViewer = displayVoxelMesh(viewer, offset, FLUID);

	auto &solidViewer = displayVoxelMesh(viewer, offset + 1, SOLID);
	solidViewer.set_colors(RowVector3d{1, 0, 0});
}

void MaCGrid::updateGrid()
{
	Region region = volData.getEnclosingRegion();
	// This three-level for loop iterates over every voxel in the volume
	// Reset all cells. Everything that's not solid is now air,
	// on layer -1.
	int32_t z, y, x;
#pragma omp parallel for private(z)
	for (z = region.getLowerZ(); z < region.getUpperZ(); z++)
#pragma omp parallel for private(y) shared(z)
		for (y = region.getLowerY(); y < region.getUpperY(); y++)
#pragma omp parallel for private(x) shared(z, y)
			for (x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				auto &cell = volData.getVoxelRef(x, y, z);
				cell.layer = -1;
				cell.idx = -1;
				cell.mask.setZero();
				if (cell.type != SOLID)
					cell.type = AIR;
			}

	fluidCells.clear();

	for (int i = 0; i < marker_particles.rows(); i++)
	{
		RowVector3d loc = marker_particles.row(i);
		RowVector3i coord;
		igl::floor(loc, coord);

		// Skip if outside of the grid:
		if (!volData.getEnclosingRegion().containsPoint(
				Vector3DInt32(coord.x(), coord.y(), coord.z())))
			continue;

		auto &cell = volData.getVoxelRef(coord.x(), coord.y(), coord.z());
		if (cell.type == FLUID)
			// Cell already added
			continue;

		if (cell.type != SOLID)
		{
			// Marker-particle on a non-solid cell means
			// this cell now contains fluid:
			cell.layer = 0;
			cell.type = FLUID;
			// Identify where in the list this cell is
			// (to speed up future processes):
			cell.mask.setConstant(1);
			cell.idx = fluidCells.size();
			fluidCells.insert(&cell);
		}
	}

	borderCells.clear();

	for (auto cell : fluidCells)
		for (int c = 0; c < 3; ++c)
		{
			auto coord = cell->coord;
			coord(c) += 1;

			if (!volData.getEnclosingRegion().containsPoint(
					Vector3DInt32(coord.x(), coord.y(), coord.z())))
				continue;
			auto &neigh = volData.getVoxelRef(coord.x(), coord.y(), coord.z());
			if (neigh.type != FLUID)
			{
				neigh.mask(c) = 1;
				borderCells.insert(&neigh);
			}
		}
}

void MaCGrid::advanceField(const double timestep)
{
	applyConvection(timestep);
	externalForces(timestep);
	// applyViscosity(timestep);
	calcPressureField(timestep);
	extrapolate();
	fixSolidCellVelocities();
}

void MaCGrid::applyConvection(const double timestep)
{
#pragma omp parallel
	for (auto cell : fluidCells)
		cell->convect(*this, timestep);

#pragma omp parallel
	for (auto cell : borderCells)
		cell->convect(*this, timestep);

#pragma omp parallel
	for (auto cell : fluidCells)
	{
		cell->u = cell->u_temp;
		cell->u_temp.setZero();
	}

#pragma omp parallel
	for (auto cell : borderCells)
	{
		cell->u = cell->u_temp;
		cell->u_temp.setZero();
	}
}

void MaCGrid::externalForces(const double timestep)
{
	const Vector3d g(0, timestep * -9.81, 0);

#pragma omp parallel
	for (auto cell : fluidCells)
		cell->u += g;

#pragma omp parallel
	for (auto cell : borderCells)
		cell->u += cell->mask.cwiseProduct(g);
}

void MaCGrid::applyViscosity(const double timestep)
{
#pragma omp parallel
	for (auto cell : fluidCells)
		cell->viscosity(*this, timestep);

#pragma omp parallel
	for (auto cell : borderCells)
		cell->viscosity(*this, timestep);

#pragma omp parallel
	for (auto cell : fluidCells)
	{
		cell->u = cell->u_temp;
		cell->u_temp.setZero();
	}

#pragma omp parallel
	for (auto cell : borderCells)
	{
		cell->u = cell->u_temp;
		cell->u_temp.setZero();
	}
}

void MaCGrid::calcPressureField(const double timestep)
{
	int size = fluidCells.size(); // TODO: fluidCells size
	VectorXd b(size);
	SparseMatrix<double> A(size, size);
	ConjugateGradient<decltype(A), Lower | Upper> cg;

	vector<Triplet<decltype(A)::Scalar>> triplets;
#pragma omp parallel
	for (auto cell : fluidCells)
	{
		int nonSolid = 0, k_air = 0;
		double divergence = 0;

		for (int j = 0; j < 3; ++j)
		{
			auto c = cell->coord;
			c(j) += 1;
			const auto &pos_v = volData.getVoxel(c.x(), c.y(), c.z());
			nonSolid -= pos_v.type != SOLID;
			k_air += pos_v.type == AIR;
			c(j) -= 2;
			const auto &neg_v = volData.getVoxel(c.x(), c.y(), c.z());
			nonSolid -= neg_v.type != SOLID;
			k_air += neg_v.type == AIR;

#pragma omp critical
			{
				if (pos_v.type == FLUID)
					triplets.emplace_back(cell->idx, pos_v.idx, 1);
				if (neg_v.type == FLUID)
					triplets.emplace_back(cell->idx, neg_v.idx, 1);
			}

			divergence += pos_v.u(j) - neg_v.u(j); // cell->mask.cwiseProduct(cell->u)(j);
		}

#pragma omp critical
		triplets.emplace_back(cell->idx, cell->idx, nonSolid);
		// double divergence = (u_xp1 ? 0 : volData.getVoxel(x + 1, y, z).u(0) - u_xm1 ? 0 : u(0)) +
		// 					(u_yp1 ? 0 : volData.getVoxel(x, y + 1, z).u(1) - u_ym1 ? 0 : u(1)) +
		// 					(u_zp1 ? 0 : volData.getVoxel(x, y, z + 1).u(2) - u_zm1 ? 0 : u(2));
		/* Modified divergence ∇·u(x,y,z) =
		 * (ux(x+1,y,z)−ux(x,y,z)) +(uy(x,y+1,z)−uy(x,y,z))+(uz(x,y,z+1)−uz(x,y,z))*/

		// TODO: Account for atmospheric pressure as soon as we have a proper density!
		b(cell->idx) = density * divergence / timestep - k_air;
	}

	A.setFromTriplets(triplets.begin(), triplets.end());

	cg.compute(A);
	VectorXd p = cg.solve(b);

#pragma omp parallel
	for (auto cell : fluidCells)
	{
		/*∇p(x,y,z) = (p(x,y,z)−p(x−1,y,z),p(x,y,z)−p(x,y−1,z),p(x,y,z)−p(x,y,z−1) )*/
		Vector3d dp;
		for (int j = 0; j < 3; ++j)
		{
			auto c = cell->coord;
			c(j) -= 1;
			const auto &neigh = volData.getVoxel(c.x(), c.y(), c.z());
			double neigh_pressure = 0;
			if (neigh.type == AIR)
				// atmospheric pressure at density 1
				neigh_pressure = 1;
			else if (neigh.idx >= 0)
				neigh_pressure = p(neigh.idx);
			c(j) -= 2;
			const auto &other = volData.getVoxel(c.x(), c.y(), c.z());
			double other_pressure = 0;
			if (other.type == AIR)
				// atmospheric pressure at density 1
				other_pressure = 1;
			else if (other.idx >= 0)
				other_pressure = p(neigh.idx);
			dp(j) = neigh_pressure - other_pressure;
		}
		cell->u -= timestep / density * dp;
	}

#pragma omp parallel
	for (auto cell : borderCells)
	{
		Vector3d dp;
		for (int j = 0; j < 3; ++j)
		{
			auto c = cell->coord;
			c(j) -= 1;
			const auto &neigh = volData.getVoxel(c.x(), c.y(), c.z());
			double neigh_pressure = 0;
			if (neigh.type == AIR)
				// atmospheric pressure at density 1
				neigh_pressure = 1;
			else if (neigh.idx >= 0)
				neigh_pressure = p(neigh.idx);
			c(j) -= 2;
			const auto &other = volData.getVoxel(c.x(), c.y(), c.z());
			double other_pressure = 0;
			if (other.type == AIR)
				// atmospheric pressure at density 1
				other_pressure = 1;
			else if (other.idx >= 0)
				other_pressure = p(neigh.idx);
			dp(j) = neigh_pressure - other_pressure;
		}
		cell->u -= cell->mask.cwiseProduct(timestep / density * dp);
	}
}

void MaCGrid::extrapolate()
{
	set<GridCell *> from(fluidCells), to;

	for (int i = 1; i < 2 /*TODO: max(2, k)*/; ++i)
	{
		for (auto cell : from)
			for (int dir = -1; dir <= 1; dir += 2)
				for (int c = 0; c < 3; ++c)
				{
					auto coord = cell->coord;
					coord(c) += dir;

					if (!volData.getEnclosingRegion().containsPoint(
							Vector3DInt32(coord.x(), coord.y(), coord.z())))
						continue;
					auto &neigh = volData.getVoxelRef(coord.x(), coord.y(), coord.z());
					if (neigh.layer == -1)
					{
						neigh.layer = i;
						to.insert(&neigh);
					}
				}
#pragma omp parallel
		for (auto cell : to)
		{
			Vector3d sumVel = Vector3d::Zero();
			int count = 0;
			for (int dir = -1; dir <= 1; dir += 2)
				for (int c = 0; c < 3; ++c)
				{
					auto coord = cell->coord;
					coord(c) += dir;

					if (!volData.getEnclosingRegion().containsPoint(
							Vector3DInt32(coord.x(), coord.y(), coord.z())))
						continue;
					const auto &neigh = volData.getVoxelRef(coord.x(), coord.y(), coord.z());
					if (neigh.layer == i - 1)
					{
						count++;
						sumVel += neigh.u;
					}
				}
			if (!count)
				throw "There must be at least one neighbor, your algorithm is broken :D";
			// cell->u /= count;
			cell->u = sumVel.cwiseProduct(Vector3d::Constant(1) - cell->mask) / count;
		}
		from.clear();
		from.swap(to);
	}
}

void MaCGrid::fixSolidCellVelocities()
{
	Region region = volData.getEnclosingRegion();
	int32_t z, y, x;
#pragma omp parallel for private(z)
	for (z = region.getLowerZ(); z < region.getUpperZ(); z++)
#pragma omp parallel for private(y) shared(z)
		for (y = region.getLowerY(); y < region.getUpperY(); y++)
#pragma omp parallel for private(x) shared(z, y)
			for (x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				auto &cell = volData.getVoxelRef(x, y, z);
				for (int c = 0; c < 3; ++c)
				{
					auto coord = cell.coord;
					coord(c) -= 1;
					auto &neigh = volData.getVoxel(coord.x(), coord.y(), coord.z());
					if (neigh.type == SOLID && cell.u(c) < 0)
						cell.u(c) = 0;
				}
			}
}

void MaCGrid::moveParticles(const double timestep)
{
	// marker_particles.rowwise()
	// marker_particles.rowwise().unaryExpr([](const Scalar &x)->Vector3d{return traceParticle(x,
	// timestep);});//template cast<typename DerivedY::Scalar >();
	int i;
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < marker_particles.rows(); i++)
		marker_particles.row(i)
			<< traceParticle(marker_particles.row(i).transpose(), timestep).transpose();
}

template <class input_t, class output_t, class step_t>
output_t rk3(std::function<output_t(input_t)> f, input_t x, step_t h)
{
	auto k_1 = f(x);
	auto k_2 = f(x + k_1 * h * 1.0 / 2);
	auto k_3 = f(x + k_2 * h * 3.0 / 4);
	return (k_1 * 2 + k_2 * 3 + k_3 * 4) * h * 1.0 / 9;
}

Vector3d MaCGrid::traceParticle(double x, double y, double z, double t) const
{
	return traceParticle({x, y, z}, t);
}

Vector3d MaCGrid::traceParticle(const Vector3d &p, double t) const
{
#if 1
	auto V = getVelocity(p);
	V = getVelocity(p + 0.5 * t * V);
	return p + t * V;
#else
	Vector3d diff =
		rk3<Vector3d, Vector3d, double>([&](Vector3d pos) { return getVelocity(pos); }, p, t);

	Vector3d vel1 = getVelocity(p);
	Vector3d vel2 = getVelocity(p + t / 2 * vel1);
	Vector3d vel3 = getVelocity(p + t * 3 / 4 * vel2);
	return p + (vel1 * 2 + vel2 * 3 + vel3 * 4) * t / 9;
#endif
}

/*inline MyFloat velXInterpolated(MyFloat x, MyFloat y) const
	{
		MyFloat v_x =
			_vel_x_front_buffer->valueInterpolated(x, y - 0.5);
		// -0.5 Due to the MAC grid structure
		return v_x;
	};*/

Vector3d MaCGrid::getVelocity(const Vector3d &pos) const
{
	return {getInterpolatedValue(pos.x(), pos.y() - 0.5, pos.z() - 0.5, 0),
			getInterpolatedValue(pos.x() - 0.5, pos.y(), pos.z() - 0.5, 1),
			getInterpolatedValue(pos.x() - 0.5, pos.y() - 0.5, pos.z(), 2)};
}

double MaCGrid::getInterpolatedValue(double x, double y, double z, int index) const
{
	double i = std::floor(x);
	double j = std::floor(y);
	double k = std::floor(z);
#if 1
	return (i + 1 - x) * (j + 1 - y) * (k + 1 - z) * volData.getVoxel(i, j, k).u(index) +
		   (x - i) * (j + 1 - y) * (k + 1 - z) * volData.getVoxel(i + 1, j, k).u(index) +
		   (i + 1 - x) * (y - j) * (k + 1 - z) * volData.getVoxel(i, j + 1, k).u(index) +
		   (x - i) * (y - j) * (k + 1 - z) * volData.getVoxel(i + 1, j + 1, k).u(index) +
		   (i + 1 - x) * (j + 1 - y) * (z - k) * volData.getVoxel(i, j, k + 1).u(index) +
		   (x - i) * (j + 1 - y) * (z - k) * volData.getVoxel(i + 1, j, k + 1).u(index) +
		   (i + 1 - x) * (y - j) * (z - k) * volData.getVoxel(i, j + 1, k + 1).u(index) +
		   (x - i) * (y - j) * (z - k) * volData.getVoxel(i + 1, j + 1, k + 1).u(index);
#else
	double x_frac = i - x;
	double y_frac = j - y;
	double z_frac = k - z;

	double value_000 = volData.getVoxel(i, j, k).u(index);
	double value_100 = volData.getVoxel(i + 1, j, k).u(index);
	double value_010 = volData.getVoxel(i, j + 1, k).u(index);
	double value_110 = volData.getVoxel(i + 1, j + 1, k).u(index);
	double value_001 = volData.getVoxel(i, j, k + 1).u(index);
	double value_101 = volData.getVoxel(i + 1, j, k + 1).u(index);
	double value_011 = volData.getVoxel(i, j + 1, k + 1).u(index);
	double value_111 = volData.getVoxel(i + 1, j + 1, k + 1).u(index);

	double value_00 = (1 - x_frac) * value_000 + x_frac * value_100;
	double value_10 = (1 - x_frac) * value_010 + x_frac * value_110;
	double value_01 = (1 - x_frac) * value_001 + x_frac * value_101;
	double value_11 = (1 - x_frac) * value_011 + x_frac * value_111;

	double value_0 = (1 - y_frac) * value_00 + y_frac * value_10;
	double value_1 = (1 - y_frac) * value_01 + y_frac * value_11;

	return (1 - z_frac) * value_0 + z_frac * value_1;
#endif
}

GridCell::GridCell() : type(AIR), layer(-1), idx(-1)
{
	coord.setConstant(INT32_MAX);
	u.setZero();
	u_temp.setZero();
}

GridCell::GridCell(const Eigen::Vector3i &_coord, const int _layer, const CellType _type)
	: coord(_coord), layer(_layer), type(_type), idx(-1)
{
	u.setZero();
	u_temp.setZero();
}

void GridCell::convect(const MaCGrid &grid, const double timestep)
{
#if 1
	u_temp = grid.traceParticle(coord.cast<double>(), -timestep);
#else
	Vector3d u_x = grid.traceParticle(coord.x(), coord.y() + 0.5, coord.z() + 0.5, -timestep);
	Vector3d u_y = grid.traceParticle(coord.x() + 0.5, coord.y(), coord.z() + 0.5, -timestep);
	Vector3d u_z = grid.traceParticle(coord.x() + 0.5, coord.y() + 0.5, coord.z(), -timestep);
	u_temp << grid.getInterpolatedValue(u_x.x(), u_x.y() - 0.5, u_x.z() - 0.5, 0),
		grid.getInterpolatedValue(u_y.x() - 0.5, u_y.y(), u_y.z() - 0.5, 1),
		grid.getInterpolatedValue(u_z.x() - 0.5, u_z.y() - 0.5, u_z.z(), 2);
#endif
	// For border cells
	u_temp = u_temp.cwiseProduct(mask);
}

void GridCell::viscosity(const MaCGrid &grid, const double timestep)
{
	int32_t x = coord.x();
	int32_t y = coord.y();
	int32_t z = coord.z();
	// TODO: only components bordering fluid are allowed to participate
	Vector3d laplacian =
		grid.volData.getVoxel(x + 1, y, z).u + grid.volData.getVoxel(x - 1, y, z).u +
		grid.volData.getVoxel(x, y + 1, z).u + grid.volData.getVoxel(x, y - 1, z).u +
		grid.volData.getVoxel(x, y, z + 1).u + grid.volData.getVoxel(x, y, z - 1).u -
		6 * grid.volData.getVoxel(x, y, z).u;
	u_temp = mask.cwiseProduct(u + timestep * grid.viscocity * laplacian);
}
