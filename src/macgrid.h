#pragma once
#include "PolyVox/RawVolume.h"
#include <igl/opengl/glfw/Viewer.h>
#include <unordered_map>
#include <vector>

enum CellType
{
	AIR = 0,
	SOLID = 1,
	FLUID = 2
};

class MaCGrid;

class GridCell
{
  public:
	Eigen::Vector3d coord;  // Coordinate in the world
	double pressure;		// Pressure on centerpoint
	Eigen::Vector3d u;		// Velocity on edges
	Eigen::Vector3d u_temp; // Temporary storage needed in updates
	MaCGrid *grid;			// Pointer to full grid
	int layer;				// layer to indicate fluid or distance to fluid
	CellType type;			// Type of cell, either fluid, solid or air

	GridCell()
	{
	}

	GridCell(const Eigen::Vector3d &_coord, MaCGrid *_grid, const int _layer, const CellType _type);

	bool operator==(const GridCell &other) const
	{
		return coord == other.coord;
	}
};

class MaCGrid
{

  public:
	PolyVox::RawVolume<GridCell> volData;
	Eigen::MatrixXd
		marker_particles; // #P by 3 matrix of marker particles used to keep track of fluid.

	double viscocity; // Kinematic viscosity
	double density;   // Density of the fluid

	double air_density = 1; // Air density(always 1)
	double p_atm = 101325;  // ~100 kPa air pressure

	double h; // Width of a gridcell
	MaCGrid(const double _h, const double _viscocity, const double _density);

	// Add marker particles in radius
	void addParticles(const Eigen::MatrixXd &positions);

	// Get cell coordinates
	Eigen::RowVector3i getCellCoordinate(Eigen::RowVector3d worldCoordinate)
	{
		return (worldCoordinate / h - Eigen::RowVector3d::Constant(0.5)).cast<int>();
	}

	// Simulate the fluid for the specified timestep
	void simulate(const double timestep);

	// Display the marker particles as spheres
	void displayFluid(igl::opengl::glfw::Viewer &viewer, const int offSet);

  private:
	// TODO: Optional dynamic timestep

	// Update the dynamic grid based on the marker particles
	void updateGrid();

	/********************************************
	 *		Advance the velocity field u		*
	 ********************************************/
	void advanceField();

	// Backwards particle trace for convection
	void applyConvection();

	// Apply external forces(gravity)
	void externalForces();

	// Apply viscosity
	void applyViscosity();

	// Calculate pressure field to satisfy incompressability
	void calcPressureField();

	// Apply pressure term
	void applyPressure();

	// Extrapolate fluid into buffer zone
	void extrapolate();

	// Fix solid cell velocities
	void fixSolidCellVelocities();

	/********************************************
	 *		Advance the marker particles		*
	 ********************************************/
	void moveParticles();
};