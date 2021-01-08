#pragma once

#include <vector>

#include <Eigen/Dense>

namespace ODESolver {

class ImplicitEuler {
  private:
  const double omega;

  public:
  ImplicitEuler(double omega) : omega(omega){};
  void solveIVP(Eigen::Vector2d& u_0, Eigen::Matrix<double, 2, Eigen::Dynamic>& u, double dt);
};

} // namespace ODESolver
