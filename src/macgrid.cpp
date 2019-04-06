#include "macgrid.h"

#include <Eigen/IterativeLinearSolvers>
#include <PolyVox/MarchingCubesSurfaceExtractor.h>
#include <PolyVox/Mesh.h>
#include <cmath>
#include <igl/floor.h>
#include <iostream>

using namespace PolyVox;
using namespace Eigen;
using namespace std;

MaCGrid::MaCGrid(const double _h, const double _viscocity, const double _density)
	: h(_h), viscocity(_viscocity), density(_density), marker_particles(0, 3),
	  volData(Region(Vector3DInt32(-100, 0, -100), Vector3DInt32(100, 100, 100)))
{
	volData.setBorderValue(GridCell(Vector3i::Zero(), this, -1, SOLID));
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

void MaCGrid::displayFluid(igl::opengl::glfw::Viewer &viewer, const int offset)
{
	CustomController controller;
	auto mesh = extractMarchingCubesMesh(&volData, volData.getEnclosingRegion(), controller);
	auto decoded = decodeMesh(mesh);
	MatrixXd V(decoded.getNoOfVertices(), 3);
	MatrixXi T(decoded.getNoOfIndices() / 3, 3);
	for (int i = 0; i < V.rows(); i++)
	{
		auto vertex = decoded.getVertex(i).position +
					  (Vector3DFloat)volData.getEnclosingRegion().getLowerCorner();
		V.row(i) << vertex.getX(), vertex.getY(), vertex.getZ();
	}
	for (int i = 0; i < decoded.getNoOfIndices(); i++)
		T(i / 3, i % 3) = (int)decoded.getIndex(i);

	RowVector3d fluidColor(0.2, 0.2, 0.8);
	auto &viewData = viewer.data_list[offset];
	viewData.clear();
	viewData.set_mesh(V, T);
	viewData.set_face_based(true);
	viewData.set_colors(fluidColor);
	viewData.show_lines = false;
}

void MaCGrid::updateGrid()
{
	Region region = volData.getEnclosingRegion();
	// This three-level for loop iterates over every voxel in the volume
	int32_t z, y, x;
#pragma omp parallel for private(z)
	for (z = region.getLowerZ(); z < region.getUpperZ(); z++)
#pragma omp parallel for private(y) shared(z)
		for (y = region.getLowerY(); y < region.getUpperY(); y++)
#pragma omp parallel for private(x) shared(z, y)
			for (x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				auto &cell = volData.getVoxelRef(x, y, z);
				if (cell.grid == nullptr)
					continue;
				cell.layer = -1;
				if (cell.type != SOLID)
					cell.type = AIR;
			}

	fluidCells.clear();

	for (int i = 0; i < marker_particles.rows(); i++)
	{
		RowVector3d loc = marker_particles.row(i);
		RowVector3i coord;
		igl::floor(loc, coord);

		if (!volData.getEnclosingRegion().containsPoint(
				Vector3DInt32(coord.x(), coord.y(), coord.z())))
			continue;

		auto &cell = volData.getVoxelRef(coord.x(), coord.y(), coord.z());
		if (cell.type == FLUID)
			continue;
		if (cell.grid == nullptr)
			cell = GridCell(coord, this, 0, FLUID);
		else if (cell.type != SOLID)
		{
			cell.layer = 0;
			cell.type = FLUID;
		}
		else
			continue;
		cell.idx = fluidCells.size();
		fluidCells.push_back(&cell);
	}
}

void MaCGrid::advanceField(const double timestep)
{
	applyConvection(timestep);
	externalForces(timestep);
	// applyViscosity(timestep);
	calcPressureField(timestep);
	applyPressure(timestep);
	extrapolate();
	fixSolidCellVelocities();
}

void MaCGrid::applyConvection(const double timestep)
{
#pragma omp parallel
	for (auto cell : fluidCells)
		cell->convect(timestep);

#pragma omp parallel
	for (auto cell : fluidCells)
		cell->u = cell->u_temp;
}

void MaCGrid::externalForces(const double timestep)
{
	const Vector3d g(0, timestep * -9.81, 0);

#pragma omp parallel
	for (auto cell : fluidCells)
		cell->u += g;
}

void MaCGrid::applyViscosity(const double timestep)
{
#pragma omp parallel
	for (auto cell : fluidCells)
		cell->viscosity(timestep);

#pragma omp parallel
	for (auto cell : fluidCells)
		cell->u = cell->u_temp;
}

void MaCGrid::calcPressureField(const double timestep)
{
	int size = fluidCells.size(); // TODO: fluidCells size
	VectorXd p(size), b(size);
	// Matrix<int, size, -1> A;
	SparseMatrix<double> A(size, size);
	ConjugateGradient<decltype(A), Lower | Upper> cg;

	vector<Triplet<decltype(A)::Scalar>> triplets;
	int i;
#pragma omp parallel for
	for (i = 0; i < size; i++)
	{
		auto &cell = *fluidCells[i];
		int nonSolid = 0, k_air = 0;
		double divergence = 0;

		for (int j = 0; j < 3; ++j)
		{
			auto c = cell.coord;
			c(j) += 1;
			const auto &pos_v = volData.getVoxel(c.x(), c.y(), c.z());
			auto pos_solid = pos_v.type == SOLID;
			c(j) -= 2;
			const auto &neg_v = volData.getVoxel(c.x(), c.y(), c.z());
			auto neg_solid = neg_v.type == SOLID;

			nonSolid -= !pos_solid + !neg_solid;

			k_air += (pos_v.type == AIR) + (neg_v.type == AIR);

#pragma omp critical
			{
				if (pos_v.type == FLUID)
					triplets.emplace_back(i, pos_v.idx, 1);
				if (neg_v.type == FLUID)
					triplets.emplace_back(i, neg_v.idx, 1);
			}

			divergence += !pos_solid * pos_v.u(j) - !neg_solid * cell.u(j);
		}

#pragma omp critical
		triplets.emplace_back(i, i, nonSolid);
		// double divergence = (u_xp1 ? 0 : volData.getVoxel(x + 1, y, z).u(0) - u_xm1 ? 0 : u(0)) +
		// 					(u_yp1 ? 0 : volData.getVoxel(x, y + 1, z).u(1) - u_ym1 ? 0 : u(1)) +
		// 					(u_zp1 ? 0 : volData.getVoxel(x, y, z + 1).u(2) - u_zm1 ? 0 : u(2));
		/* Modified divergence ∇·u(x,y,z) =
		 * (ux(x+1,y,z)−ux(x,y,z)) +(uy(x,y+1,z)−uy(x,y,z))+(uz(x,y,z+1)−uz(x,y,z))*/

		b(i) = density /* * cellWidth */ / timestep * divergence - k_air * p_atm;
	}

	A.setFromTriplets(triplets.begin(), triplets.end());

	cg.compute(A);
	p = cg.solve(b);

#pragma omp parallel for
	for (i = 0; i < size; i++)
		fluidCells[i]->pressure = p(i);
}

void MaCGrid::applyPressure(const double timestep)
{
#pragma omp parallel
	for (auto cell : fluidCells)
	{
		/*∇p(x,y,z) = (p(x,y,z)−p(x−1,y,z),p(x,y,z)−p(x,y−1,z),p(x,y,z)−p(x,y,z−1) )*/
		Vector3d dp;
		for (int j = 0; j < 3; ++j)
		{
			auto c = cell->coord;
			c(j) -= 1;
			dp(j) = cell->pressure - volData.getVoxel(c.x(), c.y(), c.z()).pressure;
		}
		cell->u -= timestep / density * dp;
		// Vector3d(cell->p - volData.getVertex(cell->coord.x() + 1, cell.corrd));
	}
}

void MaCGrid::extrapolate()
{
}

void MaCGrid::fixSolidCellVelocities()
{
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

Vector3d MaCGrid::traceParticle(double x, double y, double z, double t)
{
	Vector3d p(x, y, z);
	return traceParticle(p, t);
}

Vector3d MaCGrid::traceParticle(const Vector3d &p, double t)
{
	auto V = getVelocity(p);
	V = getVelocity(p + .5 * t * V);
	return p + t * V;
}

Vector3d MaCGrid::getVelocity(const Vector3d &pos)
{
	Vector3d V;
	V << getInterpolatedValue(pos.x(), pos.y() - 0.5, pos.z() - 0.5, 0),
		getInterpolatedValue(pos.x() - 0.5, pos.y(), pos.z() - 0.5, 1),
		getInterpolatedValue(pos.x() - 0.5, pos.y() - 0.5, pos.z(), 2);
	return V;
}

double MaCGrid::getInterpolatedValue(double x, double y, double z, int index)
{
	double i = std::floor(x);
	double j = std::floor(y);
	double k = std::floor(z);
	return (i + 1 - x) * (j + 1 - y) * (k + 1 - z) * volData.getVoxel(i, j, k).u(index) +
		   (x - i) * (j + 1 - y) * (k + 1 - z) * volData.getVoxel(i + 1, j, k).u(index) +
		   (i + 1 - x) * (y - j) * (k + 1 - z) * volData.getVoxel(i, j + 1, k).u(index) +
		   (x - i) * (y - j) * (k + 1 - z) * volData.getVoxel(i + 1, j + 1, k).u(index) +
		   (i + 1 - x) * (j + 1 - y) * (z - k) * volData.getVoxel(i, j, k + 1).u(index) +
		   (x - i) * (j + 1 - y) * (z - k) * volData.getVoxel(i + 1, j, k + 1).u(index) +
		   (i + 1 - x) * (y - j) * (z - k) * volData.getVoxel(i, j + 1, k + 1).u(index) +
		   (x - i) * (y - j) * (z - k) * volData.getVoxel(i + 1, j + 1, k + 1).u(index);
}

GridCell::GridCell() : grid(nullptr), type(AIR), layer(-1)
{
	pressure = 0;
	coord.setConstant(INT32_MAX);
	u.setZero();
	u_temp.setZero();
}

GridCell::GridCell(const Eigen::Vector3i &_coord, MaCGrid *_grid, const int _layer,
				   const CellType _type)
	: coord(_coord), grid(_grid), layer(_layer), type(_type)
{
	pressure = 0;
	u.setZero();
	u_temp.setZero();
}

void GridCell::convect(const double timestep)
{
	Vector3d u_x = grid->traceParticle(coord.x(), coord.y() + 0.5, coord.z() + 0.5, -timestep);
	Vector3d u_y = grid->traceParticle(coord.x() + 0.5, coord.y(), coord.z() + 0.5, -timestep);
	Vector3d u_z = grid->traceParticle(coord.x() + 0.5, coord.y() + 0.5, coord.z(), -timestep);
	u_temp << grid->getInterpolatedValue(u_x.x(), u_x.y() - 0.5, u_x.z() - 0.5, 0),
		grid->getInterpolatedValue(u_y.x() - 0.5, u_y.y(), u_y.z() - 0.5, 1),
		grid->getInterpolatedValue(u_z.x() - 0.5, u_z.y() - 0.5, u_z.z(), 2);
}

void GridCell::viscosity(const double timestep)
{
	int32_t x = coord.x();
	int32_t y = coord.y();
	int32_t z = coord.z();
	// TODO: only components bordering fluid are allowed to participate
	Vector3d laplacian =
		grid->volData.getVoxel(x + 1, y, z).u + grid->volData.getVoxel(x - 1, y, z).u +
		grid->volData.getVoxel(x, y + 1, z).u + grid->volData.getVoxel(x, y - 1, z).u +
		grid->volData.getVoxel(x, y, z + 1).u + grid->volData.getVoxel(x, y, z - 1).u -
		6 * grid->volData.getVoxel(x, y, z).u;
	u_temp = u + timestep * grid->viscocity * laplacian;
}

/* PSEUDOCODE
// Trace a particle from point (x, y, z) for t time using RK2.
Point traceParticle(float x, float y, float z, float t)
	Vector V = getVelocity(x, y, z);
	V = getVelocity(x+0.5*t*V.x, y+0.5*t*V.y, z+0.5*t*V.z);
	return Point(x, y, z) + t*V;

// Get the interpolated velocity at a point in space.
Vector getVelocity(float x, float y, float z)
	Vector V;
	V.x = getInterpolatedValue(x/h, y/h-0.5, z/h-0.5, 0);
	V.y = getInterpolatedValue(x/h-0.5, y/h, z/h-0.5, 1);
	V.z = getInterpolatedValue(x/h-0.5, y/h-0.5, z/h, 2);
	return V;

// Get an interpolated data value from the grid.
float getInterpolatedValue(float x, float y, float z, int index)
	int i = floor(x);
	int j = floor(y);
	int k = floor(z);
	return  (i+1-x) * (j+1-y) * (k+1-z) * cell(i, j, k).u[index] +
			(x-i) * (j+1-y) * (k+1-z) * cell(i+1, j, k).u[index] +
			(i+1-x) * (y-j) * (k+1-z) * cell(i, j+1, k).u[index] +
			(x-i) * (y-j) * (k+1-z) * cell(i+1, j+1, k).u[index] +
			(i+1-x) * (j+1-y) * (z-k) * cell(i, j, k+1).u[index] +
			(x-i) * (j+1-y) * (z-k) * cell(i+1, j, k+1).u[index] +
			(i+1-x) * (y-j) * (z-k) * cell(i, j+1, k+1).u[index] +
			(x-i) * (y-j) * (z-k) * cell(i+1, j+1, k+1).u[index];
*/
