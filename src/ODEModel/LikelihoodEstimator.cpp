#include "ODEModel/LikelihoodEstimator.h"

#include "Eigen/src/Core/Array.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"

#include <fstream>
#include <iostream>
#include <utility>

ode_model::LikelihoodEstimator::LikelihoodEstimator(Function f) : referenceFromFile(std::move(f)) {}

ode_model::LikelihoodEstimator::LikelihoodEstimator(const std::string& filename)
    : referenceFromFile(readFromFile(filename)) {}

double ode_model::LikelihoodEstimator::caluculateLogLikelihood(const Function& solution) const {
  const Eigen::VectorXd solutionTime = solution.time;

  Eigen::VectorXd const referenceInterpolated = interpolate(referenceFromFile, solutionTime);
  Eigen::VectorXd const difference = referenceInterpolated - solution.values;

  return -std::pow(difference.lpNorm<1>(), 2);
}

ode_model::Function ode_model::readFromFile(const std::string& file) {
  std::ifstream in(file);
  std::string line;

  // read first line to get metadata
  std::getline(in, line);
  const std::string delimiter = ", ";
  const std::string token1 = line.substr(0, line.find(delimiter)); // should be dt
  const std::string token2 =
      line.substr(line.find(delimiter) + delimiter.size(), line.size()); // should be N
  const double dt = std::stof(token1);
  const auto n = std::stoi(token2);

  Eigen::VectorXd const time = Eigen::ArrayXd::LinSpaced(n, 0, (n - 1) * dt);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

  size_t i = 0;
  while (std::getline(in, line)) {
    const double value = std::stof(line);
    x(i) = value;
    i++;
  }

  return {time, x};
}

Eigen::VectorXd ode_model::interpolate(const Function& f, const Eigen::VectorXd& otherTime) {
  // consistency check:
  // othertime is a subset of f.time
  assert(otherTime(0) >= f.time(0));
  assert(otherTime(otherTime.size() - 1) <= f.time(f.time.size() - 1));

  Eigen::VectorXd otherX = Eigen::VectorXd::Zero(otherTime.size());
  size_t lastIndex = 0;
  for (size_t i = 0; i < static_cast<unsigned>(otherTime.size()); i++) {
    const double tau = otherTime(i);
    // find first time, that is bigger than tau
    for (size_t j = lastIndex; j < static_cast<unsigned>(f.time.size()); j++) {
      if (f.time(j) == tau) {
        otherX(i) = f.values(j);
        break;
      } else if (f.time(j) > tau) {
        const double alpha = (tau - f.time(j - 1)) / (f.time(j) - f.time(j - 1));
        const double beta = (f.time(j) - tau) / (f.time(j) - f.time(j - 1));
        const double xi = beta * f.values(j - 1) + alpha * f.values(j);
        otherX(i) = xi;
        lastIndex = j - 1;
        break;
      }
    }
  }
  return otherX;
}
