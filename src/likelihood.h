#pragma once

#include <Eigen/Dense>

namespace ODEModel {
static constexpr double true_omega = 0.871;
static constexpr double true_a = 0.291;

double caluculateLikelihood(Eigen::Matrix<double, 1, Eigen::Dynamic>& solution);
} // namespace ODEModel
