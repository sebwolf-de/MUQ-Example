#include "ODESolver.h"

size_t ode_model::ImplicitEuler::numberOfExecutions = 0;

ode_model::ImplicitEuler::ImplicitEuler(double omega, double dt, size_t n)
    : omega(omega), dt(dt), n(n){};
void ode_model::ImplicitEuler::solveIVP(Eigen::MatrixXd& u0,
                                        std::vector<Eigen::MatrixXd>& u) const {
  assert(n == u.capacity());
  numberOfExecutions++;

  Eigen::Matrix2d a;
  a << 1, -dt, omega * omega * dt, 1;
  const Eigen::ColPivHouseholderQR<Eigen::Matrix2d> aDecomposition = a.colPivHouseholderQr();

  u.at(0) = u0;

  for (size_t i = 1; i < n; i++) {
    u.at(i) = aDecomposition.solve(u.at(i - 1));
  }
}
std::vector<Eigen::MatrixXd> ode_model::ImplicitEuler::solveIVP(Eigen::MatrixXd& u0) const {
  std::vector<Eigen::MatrixXd> result(n);
  this->solveIVP(u0, result);
  return result;
}
double ode_model::ImplicitEuler::getDt() const { return dt; }
