cmake_minimum_required(VERSION 3.10)

# set the project name
project(ODE-solver)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main src/main.cpp)
target_sources(main PRIVATE src/ODESolver.cpp
  src/ODEModel.cpp
  src/likelihood.cpp
  src/SamplingProblem.cpp
  src/Interpolation.cpp
  src/ComponentFactory.cpp)

find_package(Eigen3 REQUIRED)
target_link_libraries(main PUBLIC ${ARMADILLO_LIBRARIES})

find_package(MUQ REQUIRED)
target_link_libraries(main ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
include_directories(${MUQ_INCLUDE_DIRS})
