cmake_minimum_required(VERSION 3.1)
project(FluidSimulation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_TETGEN			 "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI  "Use OpenGL viewer with ImGui"  ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
file(GLOB SRCFILES src/*.cpp src/*.h src/*.c)
add_compile_definitions(ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
if(MSVC)
target_include_directories(${PROJECT_NAME}_bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::tetgen igl::opengl_glfw_imgui)
