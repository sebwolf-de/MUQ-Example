#ifndef ODEMODEL_FUNCTION_H
#define ODEMODEL_FUNCTION_H

#include <Eigen/Dense>
#include <vector>

namespace ode_model {

struct Function {
  Eigen::VectorXd time;
  Eigen::VectorXd values;
};

inline Function extractSolution(const std::vector<Eigen::MatrixXd>& u, double dt,
                                size_t fusedIndex) {
  const size_t numberOfTimesteps = u.size();
  Function f;
  f.time = Eigen::VectorXd::LinSpaced(numberOfTimesteps, 0, (numberOfTimesteps - 1) * dt);
  f.values = Eigen::VectorXd::Zero(numberOfTimesteps);
  for (size_t i = 0; i < numberOfTimesteps; i++) {
    f.values(i) = u.at(i)(0, fusedIndex);
  }
  return f;
}
} // namespace ode_model

#endif
