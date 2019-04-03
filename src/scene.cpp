#include "scene.h"

#include "auxfunctions.h"
#include <fstream>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readMESH.h>
#include <igl/readOFF.h>

using namespace Eigen;
using namespace std;

Mesh::Mesh(const MatrixXd &_V, const MatrixXi &_F, const MatrixXi &_T, const double density,
		   const bool _isFixed, const RowVector3d &_COM, const RowVector4d &_orientation)
	: origV(_V), F(_F), T(_T), isFixed(_isFixed), COM(_COM), orientation(_orientation)
{
	comVelocity.setZero();
	angVelocity.setZero();

	RowVector3d naturalCOM; // by the geometry of the object

	// initializes the original geometric properties (COM + IT) of the object
	naturalCOM = initStaticProperties(density);

	origV.rowwise() -= naturalCOM; // removing the natural COM of the OFF file (natural COM is
								   // never used again)

	currV.resize(origV.rows(), origV.cols());
	for (int i = 0; i < currV.rows(); i++)
		currV.row(i) << QRot(origV.row(i), orientation) + COM;

	VectorXi boundVMask(origV.rows());
	boundVMask.setZero();
	for (int i = 0; i < F.rows(); i++)
		for (int j = 0; j < 3; j++)
			boundVMask(F(i, j)) = 1;

	// cout<<"boundVMask.sum(): "<<boundVMask.sum()<<endl;

	vector<int> boundTList;
	for (int i = 0; i < T.rows(); i++)
	{
		int incidence = 0;
		for (int j = 0; j < 4; j++)
			incidence += boundVMask(T(i, j));
		if (incidence > 2)
			boundTList.push_back(i);
	}

	boundTets.resize(boundTList.size());
	for (int i = 0; i < boundTets.size(); i++)
		boundTets(i) = boundTList[i];
}

RowVector3d Mesh::initStaticProperties(const double density)
{
	tetVolumes.conservativeResize(T.rows());

	RowVector3d naturalCOM;
	naturalCOM.setZero();
	Matrix3d IT;
	IT.setZero();
	for (int i = 0; i < T.rows(); i++)
	{
		Vector3d e01 = origV.row(T(i, 1)) - origV.row(T(i, 0));
		Vector3d e02 = origV.row(T(i, 2)) - origV.row(T(i, 0));
		Vector3d e03 = origV.row(T(i, 3)) - origV.row(T(i, 0));
		Vector3d tetCentroid =
			(origV.row(T(i, 0)) + origV.row(T(i, 1)) + origV.row(T(i, 2)) + origV.row(T(i, 3))) /
			4.0;
		tetVolumes(i) = abs(e01.dot(e02.cross(e03))) / 6.0;

		naturalCOM += tetVolumes(i) * tetCentroid;
	}

	totalVolume = tetVolumes.sum();
	totalMass = density * totalVolume;
	naturalCOM.array() /= totalVolume;

	// computing inertia tensor
	for (int i = 0; i < T.rows(); i++)
	{
		RowVector4d xvec;
		xvec << origV(T(i, 0), 0) - naturalCOM(0), origV(T(i, 1), 0) - naturalCOM(0),
			origV(T(i, 2), 0) - naturalCOM(0), origV(T(i, 3), 0) - naturalCOM(0);
		RowVector4d yvec;
		yvec << origV(T(i, 0), 1) - naturalCOM(1), origV(T(i, 1), 1) - naturalCOM(1),
			origV(T(i, 2), 1) - naturalCOM(1), origV(T(i, 3), 1) - naturalCOM(1);
		RowVector4d zvec;
		zvec << origV(T(i, 0), 2) - naturalCOM(2), origV(T(i, 1), 2) - naturalCOM(2),
			origV(T(i, 2), 2) - naturalCOM(2), origV(T(i, 3), 2) - naturalCOM(2);

		double I00, I11, I22, I12, I21, I01, I10, I02, I20;
		Matrix4d sumMat = Matrix4d::Constant(1.0) + Matrix4d::Identity();
		I00 = density * 6 * tetVolumes(i) *
			  (yvec * sumMat * yvec.transpose() + zvec * sumMat * zvec.transpose()).sum() / 120.0;
		I11 = density * 6 * tetVolumes(i) *
			  (xvec * sumMat * xvec.transpose() + zvec * sumMat * zvec.transpose()).sum() / 120.0;
		I22 = density * 6 * tetVolumes(i) *
			  (xvec * sumMat * xvec.transpose() + yvec * sumMat * yvec.transpose()).sum() / 120.0;
		I12 = I21 = -density * 6 * tetVolumes(i) * (yvec * sumMat * zvec.transpose()).sum() / 120.0;
		I10 = I01 = -density * 6 * tetVolumes(i) * (xvec * sumMat * zvec.transpose()).sum() / 120.0;
		I20 = I02 = -density * 6 * tetVolumes(i) * (xvec * sumMat * yvec.transpose()).sum() / 120.0;

		Matrix3d currIT;
		currIT << I00, I01, I02, I10, I11, I12, I20, I21, I22;

		IT += currIT;
	}
	invIT = IT.inverse();

	return naturalCOM;
}

Matrix3d Mesh::getCurrInvInertiaTensor()
{
	Matrix3d R = Q2RotMatrix(orientation);
	return R * invIT * R.transpose();
}

void Mesh::integrate(const double timestep)
{
	if (isFixed)
		return;

	// TODO: Decide if we want this or keep all rigid bodies still and only colide fluid with rigid
	// body

	// Increase velocity by gravity
	RowVector3d gravity(0, -9.81, 0);
	comVelocity += gravity * timestep;

	// Position based on linear velocity
	COM += comVelocity * timestep;

	// Orientation based on angular velocity
	RowVector4d angQuaternion;
	angQuaternion << 0.0, 0.5 * angVelocity * timestep;
	orientation += QMult(angQuaternion, orientation);
	// renormalizing for stability
	orientation.normalize();

	for (int i = 0; i < currV.rows(); i++)
		currV.row(i) << QRot(origV.row(i), orientation) + COM;
}

void Scene::addMesh(const MatrixXd &V, const MatrixXi &F, const MatrixXi &T, const double density,
					const bool isFixed, const RowVector3d &COM, const RowVector4d &orientation)
{
	meshes.push_back(Mesh(V, F, T, density, isFixed, COM, orientation));
}

void Scene::updateScene(const double timeStep)
{
	// integrating velocity, position and orientation from forces and previous states
	for (int i = 0; i < meshes.size(); i++)
		meshes[i].integrate(timeStep);

	currTime += timeStep;
}

bool Scene::loadScene(const string dataFolder, const string sceneFileName)
{
	ifstream sceneFileHandle, constraintFileHandle;
	sceneFileHandle.open(dataFolder + string("/") + sceneFileName);
	if (!sceneFileHandle.is_open())
		return false;
	int numofObjects;

	currTime = 0;
	sceneFileHandle >> numofObjects;
	for (int i = 0; i < numofObjects; i++)
	{
		MatrixXi objT, objF;
		MatrixXd objV;
		string MESHFileName;
		bool isFixed;
		double youngModulus, poissonRatio, density;
		RowVector3d userCOM;
		RowVector4d userOrientation;
		sceneFileHandle >> MESHFileName >> density >> youngModulus >> poissonRatio >> isFixed >>
			userCOM(0) >> userCOM(1) >> userCOM(2) >> userOrientation(0) >> userOrientation(1) >>
			userOrientation(2) >> userOrientation(3);
		userOrientation.normalize();
		if (MESHFileName.find(".off") != std::string::npos)
		{
			MatrixXd VOFF;
			MatrixXi FOFF;
			igl::readOFF(dataFolder + std::string("/") + MESHFileName, VOFF, FOFF);
			RowVectorXd mins = VOFF.colwise().minCoeff();
			RowVectorXd maxs = VOFF.colwise().maxCoeff();
			for (int k = 0; k < VOFF.rows(); k++)
				VOFF.row(k) << 25.0 * (VOFF.row(k) - mins).array() / (maxs - mins).array();

			if (!isFixed)
				igl::copyleft::tetgen::tetrahedralize(VOFF, FOFF, "pq1.1", objV, objT, objF);
			else
				igl::copyleft::tetgen::tetrahedralize(VOFF, FOFF, "pq1.414Y", objV, objT, objF);
		}
		else
		{
			igl::readMESH(dataFolder + std::string("/") + MESHFileName, objV, objT, objF);
		}

		// fixing weird orientation problem
		MatrixXi tempF(objF.rows(), 3);
		tempF << objF.col(2), objF.col(1), objF.col(0);
		objF = tempF;

		addMesh(objV, objF, objT, density, isFixed, userCOM, userOrientation);
	}
}
