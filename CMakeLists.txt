cmake_minimum_required(VERSION 3.10)

# set the project name
project(MUQ-Example)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

option(USE_CLANG_TIDY "Use clang tidy during compilation" OFF)
if(USE_CLANG_TIDY) 
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
endif()

find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
find_package(MUQ REQUIRED)

add_library(uq 
  src/UQ/SamplingProblem.cpp
  src/UQ/StaticLoadBalancer.cpp
  src/UQ/MIInterpolation.cpp
  src/UQ/ParallelMIComponentFactory.cpp)
target_include_directories(uq PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_include_directories(uq SYSTEM PUBLIC ${MUQ_INCLUDE_DIRS})
target_include_directories(uq SYSTEM PUBLIC ${MPI_CXX_INCLUDE_PATH})

add_library(ode 
  src/ODEModel/ODESolver.cpp
  src/ODEModel/ODEPiece.cpp
  src/ODEModel/LikelihoodEstimator.cpp)
target_include_directories(ode PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_include_directories(ode SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(ode SYSTEM PUBLIC ${MUQ_INCLUDE_DIRS})

add_library(mpi
  src/MPI/MPIHelpers.cpp)
target_include_directories(mpi SYSTEM PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(mpi MPI::MPI_CXX)

add_executable(parallel src/parallel.cpp)
target_include_directories(parallel PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/)
target_include_directories(parallel SYSTEM PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_include_directories(parallel SYSTEM PUBLIC ${MUQ_INCLUDE_DIRS})
target_link_libraries(parallel uq)
target_link_libraries(parallel ode)
target_link_libraries(parallel mpi)
target_link_libraries(parallel ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
target_link_libraries(parallel MPI::MPI_CXX)

add_executable(tests tests/test_main.cpp)
target_sources(tests PRIVATE tests/ODEModel/LikelihoodEstimator.cpp)
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_LIST_DIR}/doctest)
target_link_libraries(tests ode)




