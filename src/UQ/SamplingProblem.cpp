#include "UQ/SamplingProblem.h"
#include "MPI/MPIHelpers.h"
#include "spdlog/spdlog.h"

#include <mpi.h>

#include "ODEModel/LikelihoodEstimator.h"

UQ::MySamplingProblem::MySamplingProblem(const std::shared_ptr<parcer::Communicator>& comm,
                                         const std::shared_ptr<MultiIndex>& index,
                                         const ODEModel::LikelihoodEstimator& estimator)
    : AbstractSamplingProblem(Eigen::VectorXi::Constant(1, NUM_PARAM),
                              Eigen::VectorXi::Constant(1, NUM_PARAM)),
      estimator(estimator), comm(comm), index(index) {
  const size_t global_rank = MPI::get_global_rank();
  spdlog::info("Run Sampling Problem with index {} on Rank {}.", index->GetValue(0), global_rank);
}

double UQ::MySamplingProblem::LogDensity(std::shared_ptr<SamplingState> const& state) {
  runCount++;
  lastState = state;

  const double badLikelihood = -24;
  // Discard stupid parameters
  if (state->state[0][0] > 1.0 || state->state[0][0] < 0.0 || state->state[0][1] > 1.0 ||
      state->state[0][1] < 0.0) {
    return badLikelihood;
  }

  const size_t N = std::pow(35, (index->GetValue(0) + 1)) + 1;
  auto piece = std::make_shared<ODEModel::MyODEPiece>(N);
  std::vector<Eigen::VectorXd> inputs(1);
  inputs.at(0) = state->state[0];
  std::vector<Eigen::VectorXd> outputs = piece->Evaluate(inputs);
  Eigen::Matrix<double, 1, Eigen::Dynamic> solution = outputs.at(0);
  const auto logLikelihood = estimator.caluculateLogLikelihood(solution);

  // Find out the global rank on which I am running:
  const int global_rank = MPI::get_global_rank();
  // Create some debug output
  spdlog::debug("Completed run {} on rank {}, run model for parameter: ({}, {}) with {} DOFs, "
                "likelihood: {}.",
                runCount, global_rank, state->state[0][0], state->state[0][1], N, logLikelihood);
  return logLikelihood;
}

std::shared_ptr<UQ::SamplingState> UQ::MySamplingProblem::QOI() {
  assert(lastState != nullptr);
  return std::make_shared<SamplingState>(lastState->state, 1.0);
}

size_t UQ::MySamplingProblem::runCount = 0;
