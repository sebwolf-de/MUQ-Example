cmake_minimum_required(VERSION 3.10)

# set the project name
project(MUQ-Example)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main src/main.cpp)
target_sources(main PRIVATE src/ODEModel/ODESolver.cpp
  src/ODEModel/ODEPiece.cpp
  src/ODEModel/LikelihoodEstimator.cpp
  src/UQ/SamplingProblem.cpp
  src/UQ/MIInterpolation.cpp
  src/UQ/MIComponentFactory.cpp)

include_directories(src/)

find_package(Eigen3 REQUIRED)

find_package(MUQ REQUIRED)
target_link_libraries(main ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
include_directories(${MUQ_INCLUDE_DIRS})
