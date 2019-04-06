#include "macgrid.h"

#include "PolyVox/CubicSurfaceExtractor.h"
#include "PolyVox/MarchingCubesSurfaceExtractor.h"
#include "PolyVox/Mesh.h"
#include "point_spheres.h"
#include <iostream>

using namespace PolyVox;
using namespace Eigen;
using namespace std;

void createSphereInVolume(RawVolume<GridCell> &volData, float fRadius, GridCell value)
{
	Region region = volData.getEnclosingRegion();

	// This three-level for loop iterates over every voxel in the volume
	for (int z = region.getLowerZ(); z < region.getUpperZ(); z++)
	{
		for (int y = region.getLowerY(); y < region.getUpperY(); y++)
		{
			for (int x = region.getLowerX(); x < region.getUpperX(); x++)
			{
				// Store our current position as a vector...
				Vector3DInt32 v3dCurrentPos(x, y, z);
				// And compute how far the current position is from the center of the volume
				float fDistToCenter = (v3dCurrentPos - region.getCentre()).length();

				// If the current voxel is less than 'radius' units from the center then we make it
				// solid.
				if (fDistToCenter <= fRadius)
					volData.setVoxel(x, y, z, value);
			}
		}
	}
}

MaCGrid::MaCGrid(const double _h, const double _viscocity, const double _density)
	: h(_h), viscocity(_viscocity), density(_density), marker_particles(0, 3),
	  volData(Region(Vector3DInt32(-100, 0, -100), Vector3DInt32(100, 100, 100)))
{
	GridCell test(Vector3d(0, 0, 0), this, 0, SOLID);
	createSphereInVolume(volData, 20, test);
	test.type = FLUID;
	createSphereInVolume(volData, 10, test);
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

void MaCGrid::displayFluid(igl::opengl::glfw::Viewer &viewer, const int offset)
{
	/*MatrixXd V; // Vertices of particles;
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
	viewData.show_lines = false;*/

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
