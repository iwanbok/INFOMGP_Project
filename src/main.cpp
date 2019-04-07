#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <iostream>

#include <dirent.h>

#include "macgrid.h"
#include "scene.h"

using namespace Eigen;
using namespace std;

MatrixXd V;
MatrixXi F;
igl::opengl::glfw::Viewer mgpViewer;

float currTime = 0;
float timeStep = 0.02f;

#ifndef ASSET_PATH
#define ASSET_PATH "data"
#endif // !ASSET_PATH
string dataPath(ASSET_PATH);
Scene scene;
MaCGrid grid(0.2, 8.9e-4, 1);
vector<string> sceneFiles;
int sceneID = 0;

MatrixXd platV;
MatrixXi platF;
MatrixXi platT;
RowVector3d platCOM;
RowVector4d platOrientation;

void createPlatform()
{
	double platWidth = 100.0;
	platCOM << 0.0, -5.0, -0.0;
	platV.resize(9, 3);
	platF.resize(12, 3);
	platT.resize(12, 4);
	platV << -platWidth, 0.0, -platWidth, -platWidth, 0.0, platWidth, platWidth, 0.0, platWidth,
		platWidth, 0.0, -platWidth, -platWidth, -platWidth / 10.0, -platWidth, -platWidth,
		-platWidth / 10.0, platWidth, platWidth, -platWidth / 10.0, platWidth, platWidth,
		-platWidth / 10.0, -platWidth, 0.0, -platWidth / 20.0, 0.0;
	platF << 0, 1, 2, 2, 3, 0, 6, 5, 4, 4, 7, 6, 1, 0, 5, 0, 4, 5, 2, 1, 6, 1, 5, 6, 3, 2, 7, 2, 6,
		7, 0, 3, 4, 3, 7, 4;

	platOrientation << 1.0, 0.0, 0.0, 0.0;

	platT << platF, VectorXi::Constant(12, 8);
}

class CustomMenu : public igl::opengl::glfw::imgui::ImGuiMenu
{

	virtual void draw_viewer_menu() override
	{
		// Add new group
		if (ImGui::CollapsingHeader("Algorithm Options", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::InputFloat("Time Step", &timeStep))
			{
				mgpViewer.core.animation_max_fps = (((int)1.0 / timeStep));
			}
		}
	}
};

void initializeScene()
{
	scene = Scene();
	// create platform
	createPlatform();
	scene.addMesh(platV, platF, platT, 10000.0, true, platCOM, platOrientation);
	// load scene from file
	scene.loadScene(dataPath, sceneFiles[sceneID]);
	// scene.setPlatformBarriers(platV, CRCoeff);

	currTime = 0;
	scene.updateScene(0.0);
}

void initializeMeshes()
{
	mgpViewer.selected_data_index = mgpViewer.data_list.size() - 1;
	while (mgpViewer.erase_mesh(mgpViewer.selected_data_index))
		;
	mgpViewer.data().clear();

	// Viewer Settings
	for (int i = 1; i < scene.meshes.size(); i++)
		mgpViewer.append_mesh();
}

void initializeGrid()
{
	mgpViewer.append_mesh();
	MatrixXd initialParticles(1000000, 3);
	int i = 0;
	for (int x = -50; x < 50; x++)
		for (int y = 450; y < 550; y++)
			for (int z = -50; z < 50; z++)
			{
				initialParticles.row(i) << x, y, z;
				i++;
			}
	initialParticles /= 10;
	grid.addParticles(initialParticles);
}

void updateMeshes(igl::opengl::glfw::Viewer &viewer)
{
	RowVector3d platColor;
	platColor << 0.8, 0.8, 0.8;
	RowVector3d meshColor;
	meshColor << 0.8, 0.2, 0.2;
	viewer.core.align_camera_center(scene.meshes[0].currV);
	for (int i = 0; i < scene.meshes.size(); i++)
	{
		viewer.data_list[i].clear();
		viewer.data_list[i].set_mesh(scene.meshes[i].currV, scene.meshes[i].F);
		viewer.data_list[i].set_face_based(true);
		viewer.data_list[i].set_colors(meshColor);
		viewer.data_list[i].show_lines = false;
	}
	viewer.data_list[0].show_lines = false;
	viewer.data_list[0].set_colors(platColor.replicate(scene.meshes[0].F.rows(), 1));
	viewer.data_list[0].set_face_based(true);
	// viewer.core.align_camera_center(scene.meshes[0].currV);
}

bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
	if (viewer.core.is_animating)
	{
		scene.updateScene(timeStep);
		currTime += timeStep;
		// cout <<"currTime: "<<currTime<<endl;
		updateMeshes(viewer);
		grid.simulate(timeStep);
		grid.displayFluid(viewer, scene.meshes.size());
	}

	return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
	if (key == ' ')
	{
		viewer.core.is_animating = !viewer.core.is_animating;
		if (viewer.core.is_animating)
			cout << "Simulation running" << endl;
		else
			cout << "Simulation paused" << endl;
		return true;
	}

	if (key == 'S')
	{
		if (!viewer.core.is_animating)
		{
			scene.updateScene(timeStep);
			currTime += timeStep;
			updateMeshes(viewer);
			std::cout << "currTime: " << currTime << std::endl;
			grid.simulate(timeStep);
			grid.displayFluid(viewer, scene.meshes.size());
			return true;
		}
	}

	if (sceneFiles.size() > 1 && (key == '.' || key == ','))
	{
		viewer.core.is_animating = false;
		auto nextId = sceneID + (key == '.' ? 1 : -1);
		sceneID =
			(nextId % (int)sceneFiles.size() + (int)sceneFiles.size()) % (int)sceneFiles.size();
		cout << endl << "Switching to scene " << sceneFiles[sceneID] << endl;
		initializeScene();
		initializeMeshes();
		updateMeshes(mgpViewer);
		return true;
	}

	if (key == 'R')
	{
		cout << endl << "Resetting scene" << endl;
		viewer.core.is_animating = false;
		initializeScene();
		initializeMeshes();
		updateMeshes(mgpViewer);
		return true;
	}

	return false;
}

static bool endsWith(const std::string &str, const std::string &ending)
{
	if (str.length() < ending.length())
		return false;
	return str.compare(str.length() - ending.length(), ending.length(), ending) == 0;
}

bool getScenesFromDirectory()
{
	const string sceneExtension("-scene.txt");
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(dataPath.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
			if (ent->d_type == DT_REG)
			{
				string sname(ent->d_name);
				if (endsWith(sname, sceneExtension))
					sceneFiles.push_back(sname);
			}
		closedir(dir);
	}
	else
	{
		cerr << "Failed to open directory" << endl;
		return false;
	}
	cout << "Scenes:" << endl;
	for (const auto &file : sceneFiles)
		cout << file << endl;
	if (sceneFiles.size() < 1)
	{
		cout << "No scene files found in directory. Make sure scene files end in \"-scene.txt\"."
			 << endl;
		return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	// Load scene
	if (argc > 1)
		dataPath = string(argv[1]);
	if (argc == 3)
	{
		cout << "Please provide path (argument 1) and optionally name of scene file (argument 2) "
				"and name of constraints file (argument 3)! Or use the build in scene switcher by "
				"using , and . to switch between scenes."
			 << endl;
		return EXIT_FAILURE;
	}
	else if (argc >= 4)
	{
		cout << "scene file: " << string(argv[2]) << endl;
		sceneFiles.push_back({argv[2], argv[3]});
	}
	else if (!getScenesFromDirectory())
		return EXIT_FAILURE;

	initializeScene();

	initializeMeshes();

	initializeGrid();

	mgpViewer.callback_pre_draw = &pre_draw;
	mgpViewer.callback_key_down = &key_down;
	mgpViewer.core.is_animating = false;
	mgpViewer.core.animation_max_fps = 50.;
	updateMeshes(mgpViewer);
	grid.displayFluid(mgpViewer, scene.meshes.size());
	CustomMenu menu;
	mgpViewer.plugins.push_back(&menu);

	cout << "Press [space] to toggle continuous simulation" << endl;
	cout << "Press 'S' to advance time step-by-step" << endl;
	cout << "Press 'R' to reset scene" << endl;
	if (sceneFiles.size() > 1)
		cout << "Press ',' and '.' to switch between scenes" << endl;

	mgpViewer.launch();
}