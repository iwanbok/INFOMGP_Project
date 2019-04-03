#pragma once
#include <Eigen/Core>
#include <vector>

class Mesh
{
  public:
	Eigen::MatrixXd
		origV; // original vertex positions, where COM=(0.0,0.0,0.0) - never change this!
	Eigen::MatrixXd currV; // current vertex position
	Eigen::MatrixXi F;	 // faces of the tet mesh
	Eigen::MatrixXi T;	 // Tets in the tet mesh

	Eigen::VectorXi boundTets; // indices (from T) of just the boundary tets, for collision

	// position of object in space. We must always have that currV = QRot(origV, orientation)+ COM
	Eigen::RowVector4d orientation; // current orientation
	Eigen::RowVector3d COM;			// current center of mass
	Eigen::Matrix3d invIT; // Original *inverse* inertia tensor around the COM, defined in the rest
						   // state to the object (so to the canonical world system)

	Eigen::VectorXd tetVolumes; //|T|x1 tetrahedra volumes
	Eigen::VectorXd invMasses;  //|T|x1 tetrahedra *inverse* masses

	// kinematics
	bool isFixed;	 // is the object immobile
	double totalMass; // sum(1/invMass)
	double totalVolume;
	Eigen::RowVector3d comVelocity; // the linear velocity of the center of mass
	Eigen::RowVector3d angVelocity; // the angular velocity of the object.

	Mesh(const Eigen::MatrixXd &_V, const Eigen::MatrixXi &_F, const Eigen::MatrixXi &_T,
		 const double density, const bool _isFixed, const Eigen::RowVector3d &_COM,
		 const Eigen::RowVector4d &_orientation);

	Eigen::RowVector3d initStaticProperties(const double density);

	Eigen::Matrix3d getCurrInvInertiaTensor();

	void integrate(const double timestep);
};

class Scene
{
	double currTime;

  public:
	std::vector<Mesh> meshes;

	void addMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &T,
				 const double density, const bool isFixed, const Eigen::RowVector3d &COM,
				 const Eigen::RowVector4d &orientation);

	void updateScene(const double timeStep);

	// loading a scene from the scene .txt files
	bool loadScene(const std::string dataFolder, const std::string sceneFileName);
};