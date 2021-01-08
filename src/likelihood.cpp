#include "likelihood.h"

double ODEModel::caluculateLikelihood(Eigen::Matrix<double, 1, Eigen::Dynamic>& solution) {
  const size_t N = solution.cols();

  // sample true solution in the interval [0,1]
  const Eigen::Array<double, 1, Eigen::Dynamic> omega_time =
      true_omega * Eigen::ArrayXd::LinSpaced(N, 0, 1);
  const Eigen::Matrix<double, 1, Eigen::Dynamic> true_solution = true_a * omega_time.cos();
  const Eigen::Matrix<double, 1, Eigen::Dynamic> difference = true_solution - solution;

  return -std::pow(difference.norm(), 2);
}
