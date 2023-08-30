#pragma once
#include <vector>
#include <Eigen/Dense>

namespace ODEModel {

struct Function {
  Eigen::VectorXd time;
  Eigen::VectorXd values;
};

template<size_t NUMBER_OF_FUSED_SIMS>
Function extractSolution(const std::vector<Eigen::Matrix<double, 2, NUMBER_OF_FUSED_SIMS>>& u, double dt, size_t fusedIndex) {
  const size_t NUMBER_OF_TIMESTEPS = u.size();
  Function f;
  f.time = Eigen::VectorXd::LinSpaced(NUMBER_OF_TIMESTEPS, 0, (NUMBER_OF_TIMESTEPS-1)*dt);
  f.values = Eigen::VectorXd::Zero(NUMBER_OF_TIMESTEPS);
  for (size_t i = 0; i < NUMBER_OF_TIMESTEPS; i++) {
    f.values(i) = u.at(i)(0, fusedIndex);
  }
  return f;
}
}
