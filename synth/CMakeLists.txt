cmake_minimum_required(VERSION 2.8)

project(synth)
set(CMAKE_BUILD_TYPE debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall")

add_executable(synth synth.cpp)
add_executable(viz viz.cpp)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(GLEW REQUIRED)

# Finding Eigen
# Make sure that we can find Eigen
find_package(PkgConfig QUIET)
pkg_check_modules(PC_EIGEN eigen3)
set(EIGEN_DEFINITIONS ${PC_EIGEN_CFLAGS_OTHER})
find_path(EIGEN_INCLUDE_DIRS Eigen/Core
  HINTS ${PC_EIGEN_INCLUDEDIR} ${PC_EIGEN_INCLUDE_DIRS} "${EIGEN_ROOT}" "$ENV{EIGEN_ROOT}"
  PATHS "$ENV{PROGRAMFILES}/Eigen" "$ENV{PROGRAMW6432}/Eigen"
        "$ENV{PROGRAMFILES}/Eigen 3.0.0" "$ENV{PROGRAMW6432}/Eigen 3.0.0"
  PATH_SUFFIXES eigen3 include/eigen3 include)
# Include Eigen
if(EIGEN_INCLUDE_DIRS)
  include_directories(${EIGEN_INCLUDE_DIRS})
  message(STATUS "Found Eigen: ${EIGEN_INCLUDE_DIRS}")
else(EIGEN_INCLUDE_DIRS)
  message(FATAL_ERROR "Eigen not found")
endif(EIGEN_INCLUDE_DIRS)

link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${GLEW_INCLUDE_DIRS})

target_link_libraries(synth ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(viz ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
