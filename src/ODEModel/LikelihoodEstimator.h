#pragma once

#include <Eigen/Dense>

namespace ODEModel {

struct function {
  Eigen::VectorXd time;
  Eigen::VectorXd x;
};
function readFromFile(const std::string& filename);
Eigen::VectorXd interpolate(const function& f, const Eigen::VectorXd& other_time);

class LikelihoodEstimator {
  private:
  const function reference_from_file;

  public:
  LikelihoodEstimator(const std::string& file);
  LikelihoodEstimator(const function);
  double caluculateLogLikelihood(const Eigen::Matrix<double, 1, Eigen::Dynamic>& solution) const;
};
} // namespace ODEModel
