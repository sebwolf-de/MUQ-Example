#pragma once

#include <vector>

#include <Eigen/Dense>

namespace ODESolver {

  template<size_t NUMBER_OF_FUSED_SIMS>
    class ImplicitEuler {
      private:
        const double omega;

      public:
        using Matrix_t = Eigen::Matrix<double, 2, NUMBER_OF_FUSED_SIMS>;
        ImplicitEuler(double omega) : omega(omega){};
        void solveIVP(Matrix_t& u_0, std::vector<Matrix_t>& u, double dt) const {
          const size_t N = u.capacity();

          Eigen::Matrix2d A;
          A << 1, -dt, omega * omega * dt, 1;
          Eigen::ColPivHouseholderQR<Eigen::Matrix2d> A_decomposition = A.colPivHouseholderQr();

          u.at(0) = u_0;

          for (size_t i = 1; i < N; i++) {
            u.at(i) = A_decomposition.solve(u.at(i - 1));
          }
        }
    };

} // namespace ODESolver
