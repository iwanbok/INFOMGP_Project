#include "macgrid.h"

#include <PolyVox/MarchingCubesSurfaceExtractor.h>
#include <PolyVox/Mesh.h>
#include <iostream>

using namespace PolyVox;
using namespace Eigen;
using namespace std;


MaCGrid::MaCGrid(const double _h, const double _viscocity, const double _density)
	: h(_h), viscocity(_viscocity), density(_density), marker_particles(0, 3),
	  volData(Region(Vector3DInt32(-100, 0, -100), Vector3DInt32(100, 100, 100)))
{
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
	advanceField();
	moveParticles();
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
		auto vertex = decoded.getVertex(i).position + (Vector3DFloat)volData.getEnclosingRegion().getLowerCorner();
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
	for (int z = region.getLowerZ(); z < region.getUpperZ(); z++)
		for (int y = region.getLowerY(); y < region.getUpperY(); y++)
			for (int x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				auto cell = volData.getVoxel(x, y, z);
				if (cell.grid == nullptr)
					continue;
				cell.layer = -1;
				if(cell.type != SOLID)
					cell.type = AIR;
				volData.setVoxel(x, y, z, cell);
			}

	for(int i = 0; i < marker_particles.rows(); i++)
	{
		RowVector3d coord = marker_particles.row(i);
		coord << round(coord.x()), round(coord.y()), round(coord.z());
		auto cell = volData.getVoxel(coord.x(), coord.y(), coord.z());
		if(cell.grid == nullptr)
			cell = GridCell(coord, this, 0, FLUID);
		else if(cell.type != SOLID)
		{
			cell.layer = 0;
			cell.type = FLUID;
		}
		volData.setVoxel(coord.x(), coord.y(), coord.z(), cell);
	}
}

void MaCGrid::advanceField()
{
	applyConvection();
	externalForces();
	applyViscosity();
	calcPressureField();
	applyPressure();
	extrapolate();
	fixSolidCellVelocities();
}

void MaCGrid::applyConvection()
{
}

void MaCGrid::externalForces()
{
}

void MaCGrid::applyViscosity()
{
}

void MaCGrid::calcPressureField()
{
}

void MaCGrid::applyPressure()
{
}

void MaCGrid::extrapolate()
{
}

void MaCGrid::fixSolidCellVelocities()
{
}

void MaCGrid::moveParticles()
{
}

GridCell::GridCell(const Eigen::Vector3d &_coord, MaCGrid *_grid, const int _layer,
				   const CellType _type)
	: coord(_coord), grid(_grid), layer(_layer), type(_type)
{
	pressure = 0;
	u.setZero();
	u_temp.setZero();
}
