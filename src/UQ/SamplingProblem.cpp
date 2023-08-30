#include "UQ/SamplingProblem.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>
#include <sched.h>
#include <string>
#include <unistd.h>

#include "ODEModel/LikelihoodEstimator.h"
#include "ODEModel/ODESolver.h"
#include "boost/any.hpp"
#include <iostream>
#include <vector>

#include "spdlog/spdlog.h"

size_t uq::MySamplingProblem::MySamplingProblem::runCount = 0;

uq::MySamplingProblem::MySamplingProblem(
    std::shared_ptr<MultiIndex> index, std::shared_ptr<ode_model::ODESolver> runner,
    size_t numberOfParameters,
    size_t numberOfFusedSims,
    const std::string& referenceFileName,
    std::shared_ptr<muq::Modeling::Gaussian> targetIn)
    : AbstractSamplingProblem(
          Eigen::VectorXi::Constant(1, numberOfParameters),
          Eigen::VectorXi::Constant(1, numberOfParameters)),
          runner(std::move(runner)), 
          index(index),
      likelihoodEstimator(ode_model::LikelihoodEstimator(referenceFileName)),
      numberOfParameters(numberOfParameters), numberOfFusedSims(numberOfFusedSims),
      target(std::move(targetIn)) {
  spdlog::info("Run Sampling Problem with index {}", index->GetValue(0));
}

Eigen::VectorXd uq::MySamplingProblem::GradLogDensity(std::shared_ptr<SamplingState> const& state,
                                                      [[maybe_unused]] unsigned blockWrt) {
  return target->GradLogDensity(0, state->state); // 0 instead of wrt
}

double uq::MySamplingProblem::LogDensity(std::shared_ptr<SamplingState> const& state) {
  lastState = state;
  // prepare initial condition
  Eigen::MatrixXd u0(numberOfParameters, numberOfFusedSims);
  for (size_t i = 0; i < numberOfParameters; i++) {
    for (size_t j = 0; j < numberOfFusedSims; j++) {
      u0(i,j) = state->state.at(j)(i);
    }
  }

  std::vector<double> logDensityArray(numberOfFusedSims);
  spdlog::info("######################");
  spdlog::info("Running ODESolver on index {}", index->GetValue(0));
  const std::time_t startTime = std::time(nullptr);
  auto u = runner->solveIVP(u0);
  const std::time_t endTime = std::time(nullptr);
  const auto duration = endTime - startTime;
  spdlog::info("Executed ODESolver successfully {} times, took {} seconds.", runCount, duration);
  for (size_t i = 0; i < numberOfFusedSims; i++) {
    const auto f = ode_model::extractSolution(u, runner->getDt(), i);
    const auto likelihood = likelihoodEstimator.calculateLogLikelihood(f);
    logDensityArray.at(i) = likelihood;
  }
  state->meta["LogTarget"] = logDensityArray;
  runCount += 1;
  return logDensityArray.at(0);
}

std::shared_ptr<uq::SamplingState> uq::MySamplingProblem::QOI() {
  assert(lastState != nullptr);
  return std::make_shared<SamplingState>(lastState->state, 1.0);
}
