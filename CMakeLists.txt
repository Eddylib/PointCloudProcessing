cmake_minimum_required(VERSION 3.15)
project(NNTrees)

set(CMAKE_CXX_STANDARD 17)


aux_source_directory(src SRC)
include_directories(include)
add_executable(Trees main.cpp ${SRC})