cmake_minimum_required(VERSION 3.10)

# set the project name
project(MUQ-Example)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(uq 
  src/UQ/SamplingProblem.cpp
  src/UQ/MIInterpolation.cpp
  src/UQ/MIComponentFactory.cpp)
add_library(ode 
  src/ODEModel/ODESolver.cpp
  src/ODEModel/ODEPiece.cpp
  src/ODEModel/LikelihoodEstimator.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main uq)
target_link_libraries(main ode)

add_executable(tests tests/test_main.cpp)
target_sources(tests PRIVATE tests/ODEModel/LikelihoodEstimator.cpp)
target_link_libraries(tests ode)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/doctest)

find_package(Eigen3 REQUIRED)

find_package(MUQ REQUIRED)
target_link_libraries(main ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
include_directories(${MUQ_INCLUDE_DIRS})
