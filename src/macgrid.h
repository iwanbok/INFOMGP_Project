#pragma once
/*Based on article https://pdfs.semanticscholar.org/9d47/1060d6c48308abcc98dbed850a39dbfea683.pdf */
#include <PolyVox/RawVolume.h>
#include <igl/opengl/glfw/Viewer.h>
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
	double pressure;		// Pressure on centerpoint
	Eigen::Vector3d u;		// Velocity on edges
	Eigen::Vector3d u_temp; // Temporary storage needed in updates
	MaCGrid *grid;			// Pointer to full grid
	int layer;				// layer to indicate fluid or distance to fluid
	CellType type;			// Type of cell, either fluid, solid or air
	int idx;

	GridCell();

	GridCell(const Eigen::Vector3i &_coord, MaCGrid *_grid, const int _layer, const CellType _type);

	void convect(const double timestep);

	void viscosity(const double timestep);

	bool operator==(const GridCell &other) const
	{
		return coord == other.coord;
	}
};

class MaCGrid
{

  public:
	PolyVox::RawVolume<GridCell> volData;
	std::vector<GridCell *> fluidCells;
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

	Eigen::Vector3d traceParticle(const Eigen::Vector3d &pos, double t);

	Eigen::Vector3d traceParticle(double x, double y, double z, double t);

	Eigen::Vector3d getVelocity(const Eigen::Vector3d &pos);

	double getInterpolatedValue(double x, double y, double z, int index);

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

	// Apply external forces(gravity)
	void externalForces(const double timestep);

	// Apply viscosity
	void applyViscosity(const double timestep);

	// Calculate pressure field to satisfy incompressability
	void calcPressureField(const double timestep);

	// Apply pressure term
	void applyPressure(const double timestep);

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
  public:
	/// Used to inform the MarchingCubesSurfaceExtractor about which type it should use for
	/// representing densities.
	typedef CellType DensityType;
	/// Used to inform the MarchingCubesSurfaceExtractor about which type it should use for
	/// representing materials.
	typedef CellType MaterialType;

	/**
	 * Constructor
	 *
	 * This version of the constructor takes no parameters and sets the threshold to the middle of
	 * the representable range of the underlying type. For example, if the voxel type is 'uint8_t'
	 * then the representable range is 0-255, and the threshold will be set to 127. On the other
	 * hand, if the voxel type is 'float' then the representable range is -FLT_MAX to FLT_MAX and
	 * the threshold will be set to zero.
	 */
	CustomController(void)
	{
		m_tThreshold = FLUID;
	}

	/**
	 * Converts the underlying voxel type into a density value.
	 *
	 * The default implementation of this function just returns the voxel type directly and is
	 * suitable for primitives types. Specialisations of this class can modify this behaviour.
	 */
	DensityType convertToDensity(GridCell voxel)
	{
		return voxel.type;
	}

	/**
	 * Converts the underlying voxel type into a material value.
	 *
	 * The default implementation of this function just returns the constant '1'. There's not much
	 * else it can do, as it needs to work with primitive types and the actual value of the type is
	 * already being considered to be the density. Specialisations of this class can modify this
	 * behaviour.
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

	void setThreshold(DensityType tThreshold)
	{
		m_tThreshold = tThreshold;
	}

  private:
	DensityType m_tThreshold;
};