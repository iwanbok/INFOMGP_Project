cmake_minimum_required(VERSION 3.1)
project(FluidSimulation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_TETGEN            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use OpenGL viewer with ImGui"  ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

# Add your project files
file(GLOB SRCFILES src/*.cpp src/*.h src/*.c)
add_compile_definitions(ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")
add_executable(${PROJECT_NAME}_bin ${SRCFILES})

if(MSVC)
target_include_directories(${PROJECT_NAME}_bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

target_include_directories(${PROJECT_NAME}_bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libpolyvox/include")
target_link_libraries(${PROJECT_NAME}_bin PRIVATE igl::core igl::opengl_glfw igl::tetgen igl::opengl_glfw_imgui ${OpenMP_LIBRARIES})
