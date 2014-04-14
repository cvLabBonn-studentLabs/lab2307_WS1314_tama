cmake_minimum_required(VERSION 2.8)

project(main)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
set(CMAKE_BUILD_TYPE debug)

add_executable(main main.cpp)

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED COMPONENTS common io octree kdtree features search)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS ${PCL_LIBRARIES})
# target_link_libraries(main ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_KDTREE_LIBRARIES})
target_link_libraries(main ${OpenCV_LIBS} ${PCL_LIBRARIES})
