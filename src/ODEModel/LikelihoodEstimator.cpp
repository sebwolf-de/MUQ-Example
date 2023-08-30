#include "ODEModel/LikelihoodEstimator.h"

#include "Eigen/src/Core/Array.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"

#include <fstream>
#include <iostream>
#include <utility>

ODEModel::LikelihoodEstimator::LikelihoodEstimator(Function f)
    : referenceFromFile(std::move(f)) {}

ODEModel::LikelihoodEstimator::LikelihoodEstimator(const std::string& filename)
    : referenceFromFile(readFromFile(filename)) {}

double ODEModel::LikelihoodEstimator::caluculateLogLikelihood(
    const Function& solution) const {
  const Eigen::VectorXd solutionTime = solution.time;

  Eigen::VectorXd referenceInterpolated = interpolate(referenceFromFile, solutionTime);
  Eigen::VectorXd difference = referenceInterpolated - solution.values;

  return -std::pow(difference.lpNorm<1>(), 2);
}

ODEModel::Function ODEModel::readFromFile(const std::string& file) {
  std::ifstream in(file);
  std::string line;

  // read first line to get metadata
  std::getline(in, line);
  const std::string delimiter = ", ";
  const std::string token1 = line.substr(0, line.find(delimiter)); // should be dt
  const std::string token2 =
      line.substr(line.find(delimiter) + delimiter.size(), line.size()); // should be N
  const double dt = std::stof(token1);
  const auto N = std::stoi(token2);

  Eigen::VectorXd time = Eigen::ArrayXd::LinSpaced(N, 0, (N - 1) * dt);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(N);

  size_t i = 0;
  while (std::getline(in, line)) {
    const double value = std::stof(line);
    x(i) = value;
    i++;
  }

  return {time, x};
}

Eigen::VectorXd ODEModel::interpolate(const Function& f, const Eigen::VectorXd& otherTime) {
  // consistency check:
  // othertime is a subset of f.time
  assert(otherTime(0) >= f.time(0));
  assert(otherTime(otherTime.size() - 1) <= f.time(f.time.size() - 1));

  Eigen::VectorXd other_x = Eigen::VectorXd::Zero(otherTime.size());
  size_t last_index = 0;
  for (size_t i = 0; i < static_cast<unsigned>(otherTime.size()); i++) {
    const double tau = otherTime(i);
    // find first time, that is bigger than tau
    for (size_t j = last_index; j < static_cast<unsigned>(f.time.size()); j++) {
      if (f.time(j) == tau) {
        other_x(i) = f.values(j);
        break;
      } else if (f.time(j) > tau) {
        const double alpha = (tau - f.time(j - 1)) / (f.time(j) - f.time(j - 1));
        const double beta = (f.time(j) - tau) / (f.time(j) - f.time(j - 1));
        const double xi = beta * f.values(j - 1) + alpha * f.values(j);
        other_x(i) = xi;
        last_index = j - 1;
        break;
      }
    }
  }
  return other_x;
}
