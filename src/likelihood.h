#pragma once

#include <Eigen/Dense>

namespace ODEModel {
struct function {
    Eigen::VectorXd time;
    Eigen::VectorXd x;
};
double caluculateLogLikelihood(Eigen::Matrix<double, 1, Eigen::Dynamic>& solution);
function readFromFile(std::string file);
Eigen::VectorXd interpolate(const function& f, const Eigen::VectorXd& other_time);
} // namespace ODEModel
