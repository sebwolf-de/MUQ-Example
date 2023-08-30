#pragma once

#include <string>

#include <Eigen/Dense>

#include "Function.h"

namespace ODEModel {
Function readFromFile(const std::string& filename);
Eigen::VectorXd interpolate(const Function& f, const Eigen::VectorXd& otherTime);

class LikelihoodEstimator {
  private:
  const Function referenceFromFile;

  public:
  LikelihoodEstimator(const std::string& file);
  LikelihoodEstimator(const Function);
  double caluculateLogLikelihood(const Function& solution) const;
};
} // namespace ODEModel
