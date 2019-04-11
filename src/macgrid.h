#pragma once
/*Based on article https://pdfs.semanticscholar.org/9d47/1060d6c48308abcc98dbed850a39dbfea683.pdf */
#include <PolyVox/RawVolume.h>
#include <igl/opengl/glfw/Viewer.h>
#include <set>
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
	Eigen::Vector3i coord;  // Coordinate in the world
	Eigen::Vector3d u;		// Velocity on edges
	Eigen::Vector3d u_temp; // Temporary storage needed in updates
	Eigen::Vector3d mask;   // mask for velocity component updates
	int layer;				// layer to indicate fluid or distance to fluid
	CellType type;			// Type of cell, either fluid, solid or air
	int idx;				// Id associated with fluid cells
	double p, p2, div;		// Values needed for pressure calculation
	GridCell();

	GridCell(const Eigen::Vector3i &_coord, const int _layer, const CellType _type);

	// Use advection to determine the new velocity components
	void convect(const MaCGrid &grid, const double timestep);

	// NOT USED update the velocity using viscosity
	void viscosity(const MaCGrid &grid, const double timestep);

	bool operator==(const GridCell &other) const
	{
		return coord == other.coord;
	}
};

class MaCGrid
{

  public:
	PolyVox::RawVolume<GridCell> volData; // Voxel volume of the grid
	std::set<GridCell *> fluidCells;	  // Set of cells which are marked fluid
	std::set<GridCell *>
		borderCells; // Set of cells which contain velocity components bordering fluid
	Eigen::MatrixXd
		marker_particles; // #P by 3 matrix of marker particles used to keep track of fluid.

	Eigen::MatrixXd pV; // Particle vertices
	Eigen::MatrixXi pT; // Particle triangles

	double viscocity; // Kinematic viscosity
	double density;   // Density of the fluid

	double air_density = 1; // Air density(always 1)
	double p_atm = 1;
	// 101325;  // ~100 kPa air pressure

	double h; // Width of a gridcell(Not used)
	MaCGrid(const double _h, const double _viscocity, const double _density);

	// Reset the grid to a clean state, use this instead of calling the constructor again as the =
	// operator is deleted.
	void reset();

	// Add marker particles in radius
	void addParticles(const Eigen::MatrixXd &positions);

	// Simulate the fluid for the specified timestep
	void simulate(const double timestep);

	/// Display the voxel mesh for \p voxelType
	igl::opengl::ViewerData &displayVoxelMesh(igl::opengl::glfw::Viewer &viewer, const int offset,
											  CellType voxelType);

	// Display the marker particles as spheres
	// (Could be a const method, except that extractMarchingCubesMesh
	//  takes volData as non-const pointer).
	void displayFluid(igl::opengl::glfw::Viewer &viewer, const int offSet);

	// Trace a particle at position pos for t time
	Eigen::Vector3d traceParticle(const Eigen::Vector3d &pos, double t) const;

	// Trace a particle from x, y, z position for t time
	Eigen::Vector3d traceParticle(double x, double y, double z, double t) const;

	// Gets the interpolated velocity of the given pos
	Eigen::Vector3d getVelocity(const Eigen::Vector3d &pos) const;

	// Get a intepolated value at x, y, z position at index of u vector in gridcells
	double getInterpolatedValue(double x, double y, double z, int index) const;

  private:
	// TODO: Optional dynamic timestep

	// Update the dynamic grid based on the marker particles
	void updateGrid();

	/********************************************
	 *		Advance the velocity field u		*
	 ********************************************/
	void advanceField(const double timestep);

	// Backwards particle trace for convection
	void applyConvection(const double timestep);

	// Apply external forces (gravity)
	void externalForces(const double timestep);

	// Apply viscosity
	void applyViscosity(const double timestep);

	// Calculate pressure field to satisfy incompressability
	void calcPressureField(const double timestep);

	// Extrapolate fluid into buffer zone
	void extrapolate();

	// Fix solid cell velocities
	void fixSolidCellVelocities();

	/********************************************
	 *		Advance the marker particles		*
	 ********************************************/
	void moveParticles(const double timestep);
};

class CustomController
{
	CellType which;

  public:
	/// Used to inform the MarchingCubesSurfaceExtractor about which type it should use for
	/// representing densities.
	typedef int DensityType;
	/// Used to inform the MarchingCubesSurfaceExtractor about which type it should use for
	/// representing materials.
	typedef CellType MaterialType;

	/**
	 * Constructor
	 *
	 * This version of the constructor sets the controller to which type it should accept when
	 * making the mesh.
	 */
	CustomController(CellType filter)
	{
		which = filter;
		m_tThreshold = 1;
	}

	/**
	 * Converts the underlying voxel type into a density value.
	 */
	DensityType convertToDensity(GridCell voxel)
	{
		return voxel.type == which;
	}

	/**
	 * Converts the underlying voxel type into a material value.
	 */
	MaterialType convertToMaterial(GridCell voxel)
	{
		return voxel.type;
	}

	/**
	 * Returns a material which is in some sense a weighted combination of the supplied materials.
	 *
	 * The Marching Cubes algotithm generates vertices which lie between voxels, and ideally the
	 * material of the vertex should be interpolated from the materials of the voxels. In practice,
	 * that material type is often an integer identifier (e.g. 1 = rock, 2 = soil, 3 = grass) and an
	 * interpolation doean't make sense (e.g. soil is not a combination or rock and grass).
	 * Therefore this default interpolation just returns whichever material is associated with a
	 * voxel of the higher density, but if more advanced voxel types do support interpolation then
	 * it can be implemented in this function.
	 */
	GridCell blendMaterials(GridCell a, GridCell b, float /*weight*/)
	{
		if (convertToDensity(a) > convertToDensity(b))
			return a;
		return b;
	}

	/**
	 * Returns the density value which was passed to the constructor.
	 *
	 * As mentioned in the class description, the extracted surface will pass through the density
	 * value specified by the threshold, and so you should make sure that the threshold value you
	 * choose is between the minimum and maximum values found in your volume data. By default it is
	 * in the middle of the representable range of the underlying type.
	 */
	DensityType getThreshold(void)
	{
		return m_tThreshold;
	}

	void setThreshold(CellType _which)
	{
		which = _which;
	}

  private:
	DensityType m_tThreshold;
};
