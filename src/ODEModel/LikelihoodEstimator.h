#ifndef ODEMODEL_LIKELIHOODESTIMATOR_H
#define ODEMODEL_LIKELIHOODESTIMATOR_H

#include <string>

#include <Eigen/Dense>

#include "Function.h"

namespace ode_model {
Function readFromFile(const std::string& filename);
Eigen::VectorXd interpolate(const Function& f, const Eigen::VectorXd& otherTime);

class LikelihoodEstimator {
  private:
  const Function referenceFromFile;

  public:
  LikelihoodEstimator(const std::string& file);
  LikelihoodEstimator(Function);
  [[nodiscard]] double calculateLogLikelihood(const Function& solution) const;
};
} // namespace ode_model

#endif
