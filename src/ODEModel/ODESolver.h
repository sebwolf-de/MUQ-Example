#ifndef ODEMODEL_ODESOLVER_H
#define ODEMODEL_ODESOLVER_H

#include <cassert>
#include <vector>

#include <Eigen/Dense>

namespace ode_model {

class ODESolver {
  public:
  virtual ~ODESolver() = default;
  [[nodiscard]] virtual double getDt() const = 0;
  virtual void solveIVP(Eigen::MatrixXd& u0, std::vector<Eigen::MatrixXd>& u) const = 0;
  virtual std::vector<Eigen::MatrixXd> solveIVP(Eigen::MatrixXd& u0) const = 0;
};

class ImplicitEuler : public ODESolver {
  private:
  const double omega;
  const double dt;
  const size_t n;

  public:
  static size_t numberOfExecutions;
  ImplicitEuler(double omega, double dt, size_t n);
  void solveIVP(Eigen::MatrixXd& u0, std::vector<Eigen::MatrixXd>& u) const override;
  std::vector<Eigen::MatrixXd> solveIVP(Eigen::MatrixXd& u0) const override;
  [[nodiscard]] double getDt() const override;
};

} // namespace ode_model

#endif
