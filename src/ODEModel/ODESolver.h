#ifndef ODEMODEL_ODESOLVER_H
#define ODEMODEL_ODESOLVER_H

#include <vector>

#include <Eigen/Dense>

namespace ode_model {

template <size_t NUMBER_OF_FUSED_SIMS> class ImplicitEuler {
  private:
  const double omega;

  public:
  using Matrix_t = Eigen::Matrix<double, 2, NUMBER_OF_FUSED_SIMS>;
  ImplicitEuler(double omega) : omega(omega){};
  void solveIVP(Matrix_t& u0, std::vector<Matrix_t>& u, double dt) const {
    const size_t n = u.capacity();

    Eigen::Matrix2d a;
    a << 1, -dt, omega * omega * dt, 1;
    Eigen::ColPivHouseholderQR<Eigen::Matrix2d> aDecomposition = a.colPivHouseholderQr();

    u.at(0) = u0;

    for (size_t i = 1; i < n; i++) {
      u.at(i) = aDecomposition.solve(u.at(i - 1));
    }
  }
};

} // namespace ode_model

#endif
