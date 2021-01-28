#include "ODEModel/ODESolver.h"

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/QR/ColPivHouseholderQR.h"

void ODESolver::ImplicitEuler::solveIVP(Eigen::Vector2d& u_0,
                                        Eigen::Matrix<double, 2, Eigen::Dynamic>& u, double dt) {
  const size_t N = u.cols();

  Eigen::Matrix2d A;
  A << 1, -dt, omega * omega * dt, 1;
  Eigen::ColPivHouseholderQR<Eigen::Matrix2d> A_decomposition = A.colPivHouseholderQr();

  u.block<2, 1>(0, 0) = u_0;

  for (size_t i = 1; i < N; i++) {
    u.block<2, 1>(0, i) = A_decomposition.solve(u.block<2, 1>(0, i - 1));
  }
}
