cmake_minimum_required(VERSION 3.10)
project(NNTrees)

set(CMAKE_CXX_STANDARD 17)


aux_source_directory(src SRC)
include_directories(include)

add_library(eddyknn ${SRC})
target_include_directories(eddyknn PUBLIC include)

add_executable(test_t main.cpp)
target_link_libraries(test_t PRIVATE eddyknn)

add_executable(benchmark_knn benchmark_my_kdtree_octree.cpp)
target_link_libraries(benchmark_knn PRIVATE eddyknn)

if(${USE_ROS})
install(TARGETS eddyknn DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
endif()
