#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include <Eigen/Dense>
#include "ODESolver.h"
#include "LikelihoodEstimator.h"
#include "Function.h"

double getRandom(double min=0.0, double max=1.0) { 
  const int randomVariable = std::rand();
  const double randomVariableZeroOne = static_cast<double>(randomVariable) / RAND_MAX;
  return min + (max - min) * randomVariableZeroOne;
}

int main() {
  const size_t NUMBER_OF_FUSED_SIMS = 2;
  const double omega = 1.0;
  const double dt = 0.1;
  const size_t NUMBER_OF_TIMESTEPS = 11;

  std::srand(std::time(nullptr));

  using Matrix_t = Eigen::Matrix<double, 2, NUMBER_OF_FUSED_SIMS>;
  const auto solver = ODESolver::ImplicitEuler<NUMBER_OF_FUSED_SIMS>(omega);
  auto u = std::vector<Matrix_t>(NUMBER_OF_TIMESTEPS);
  Matrix_t u0;
  u0 << getRandom(), getRandom(), getRandom(), getRandom();
  solver.solveIVP(u0, u, dt);

  const auto likelihoodEstimator = ODEModel::LikelihoodEstimator("../data/true_solution.dat");
  for (size_t i = 0; i < NUMBER_OF_FUSED_SIMS; i++) {
    const auto f = ODEModel::extractSolution<NUMBER_OF_FUSED_SIMS>(u, dt, i);
    const auto likelihood = likelihoodEstimator.caluculateLogLikelihood(f);
    std::cout << "ODE Model with initial condition (" << u0(0,i) << ", " << u0(1,i) << ") resulted in a log-likelihood of " << likelihood << std::endl;
  }
}
