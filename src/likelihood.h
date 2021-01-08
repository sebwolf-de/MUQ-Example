#pragma once

#include <Eigen/Dense>

namespace ODEModel {
double caluculateLikelihood(Eigen::Matrix<double, 1, Eigen::Dynamic>& solution);
} // namespace ODEModel
