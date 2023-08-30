#ifndef ODEMODEL_ODESOLVER_H
#define ODEMODEL_ODESOLVER_H

#include <cassert>
#include <vector>

#include <Eigen/Dense>

namespace ode_model {

class ODESolver {
  public:
  virtual double getDt() const = 0;
  virtual void solveIVP(Eigen::MatrixXd& u0, std::vector<Eigen::MatrixXd>& u) const = 0;
  virtual std::vector<Eigen::MatrixXd> solveIVP(Eigen::MatrixXd& u0) const = 0;
};

class ImplicitEuler : public ODESolver{
  private:
  const double omega;
  const double dt;
  const size_t n;
  public:
  ImplicitEuler(double omega, double dt, size_t n) : omega(omega), dt(dt), n(n){};
  virtual void solveIVP(Eigen::MatrixXd& u0, std::vector<Eigen::MatrixXd>& u) const override {
    assert(n == u.capacity());

    Eigen::Matrix2d a;
    a << 1, -dt, omega * omega * dt, 1;
    Eigen::ColPivHouseholderQR<Eigen::Matrix2d> aDecomposition = a.colPivHouseholderQr();

    u.at(0) = u0;

    for (size_t i = 1; i < n; i++) {
      u.at(i) = aDecomposition.solve(u.at(i - 1));
    }
  }
  virtual std::vector<Eigen::MatrixXd> solveIVP(Eigen::MatrixXd& u0) const override {
    std::vector<Eigen::MatrixXd> result(n);
    this->solveIVP(u0, result);
    return result;
  }
  virtual double getDt() const override {
    return dt;
  }
};

} // namespace ode_model

#endif
