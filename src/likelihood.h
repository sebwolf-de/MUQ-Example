#pragma once

#include <Eigen/Dense>

namespace ODEModel {
double caluculateLogLikelihood(Eigen::Matrix<double, 1, Eigen::Dynamic>& solution);
} // namespace ODEModel
