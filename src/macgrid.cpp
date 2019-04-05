#include "macgrid.h"

#include "point_spheres.h"
#include <iostream>

using namespace Eigen;
using namespace std;

MaCGrid::MaCGrid(const double _h, const double _viscocity, const double _density)
	: h(_h), viscocity(_viscocity), density(_density), marker_particles(0, 3)
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
	// updateGrid();
	advanceField();
	moveParticles();
}

void MaCGrid::displayFluid(igl::opengl::glfw::Viewer &viewer, const int offset)
{
	MatrixXd V; // Vertices of particles;
	MatrixXi T; // Triangles of particles;
	MatrixXd C; // Colors of particles;

	RowVector3d fluidColor(0.2, 0.2, 0.8);
	directional::point_spheres(marker_particles, h,
							   fluidColor.replicate(marker_particles.rows(), 1), 10, V, T, C);
	auto &viewData = viewer.data_list[offset];
	viewData.clear();
	viewData.set_mesh(V, T);
	viewData.set_face_based(true);
	viewData.set_colors(C);
	viewData.show_lines = false;
}

void MaCGrid::updateGrid()
{
	for (auto &c : dynamicGrid)
		c.second.layer = -1;

	for (int i = 0; i < marker_particles.rows(); i++)
	{
		RowVector3i loc = getCellCoordinate(marker_particles.row(i));
		if (dynamicGrid.find(loc) == dynamicGrid.end())
			dynamicGrid[loc] = GridCell(loc.cast<double>() * h, this, 0, FLUID);
		else if (dynamicGrid[loc].type != SOLID)
		{
			dynamicGrid[loc].type = FLUID;
			dynamicGrid[loc].layer = 0;
		}
	}

	for (int i = 1; i <= 2 /*max(2, ceiling(k_cfl))*/; i++)
	{
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
