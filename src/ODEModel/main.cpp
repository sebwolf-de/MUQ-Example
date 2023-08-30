#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "Function.h"
#include "LikelihoodEstimator.h"
#include "ODESolver.h"
#include <Eigen/Dense>

double getRandom(double min = 0.0, double max = 1.0) {
  const int randomVariable = std::rand();
  const double randomVariableZeroOne = static_cast<double>(randomVariable) / RAND_MAX;
  return min + (max - min) * randomVariableZeroOne;
}

int main() {
  const size_t numberOfFusedSims = 4;
  const size_t numberOfParameters = 2;
  const double omega = 1.0;
  const double dt = 0.1;
  const size_t numberOfTimesteps = 11;

  std::srand(std::time(nullptr));

  const auto solver = ode_model::ImplicitEuler(omega, dt, numberOfTimesteps);
  auto u = std::vector<Eigen::MatrixXd>(numberOfTimesteps);
  Eigen::MatrixXd u0(numberOfParameters, numberOfFusedSims);
  for (size_t i = 0; i < numberOfParameters; i++) {
    for (size_t j = 0; j < numberOfFusedSims; j++) {
      u0(i, j) = getRandom();
    }
  }
  solver.solveIVP(u0, u);

  const auto likelihoodEstimator = ode_model::LikelihoodEstimator("../data/true_solution.dat");
  for (size_t i = 0; i < numberOfFusedSims; i++) {
    const auto f = ode_model::extractSolution(u, dt, i);
    const auto likelihood = likelihoodEstimator.calculateLogLikelihood(f);
    std::cout << "ODE Model with initial condition (" << u0(0, i) << ", " << u0(1, i)
              << ") resulted in a log-likelihood of " << likelihood << std::endl;
  }
}
