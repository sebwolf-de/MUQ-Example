#include "UQ/ParallelMIComponentFactory.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include "UQ/MIInterpolation.h"
#include "UQ/SamplingProblem.h"

#include <memory>
#include <utility>

std::shared_ptr<UQ::MCMCProposal> UQ::MyParallelMIComponentFactory::Proposal(
    [[maybe_unused]] const std::shared_ptr<MultiIndex>& index,
    const std::shared_ptr<AbstractSamplingProblem>& samplingProblem) {
  pt::ptree pt;
  pt.put("BlockIndex", 0);

  auto mu = Eigen::VectorXd::Zero(NUM_PARAM);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(NUM_PARAM, NUM_PARAM);
  // covariance is due to some hyperparameter tuning
  constexpr double covariance = 0.005;
  cov *= covariance;

  auto prior = std::make_shared<Gaussian>(mu, cov);

  return std::make_shared<MHProposal>(pt, samplingProblem, prior);
}

std::shared_ptr<UQ::MultiIndex> UQ::MyParallelMIComponentFactory::FinestIndex() {
  auto index = std::make_shared<MultiIndex>(1);
  index->SetValue(0, 0);
  return index;
}

std::shared_ptr<UQ::MCMCProposal> UQ::MyParallelMIComponentFactory::CoarseProposal(
    [[maybe_unused]] std::shared_ptr<MultiIndex> const& fineIndex,
    std::shared_ptr<MultiIndex> const& coarseIndex,
    std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
    std::shared_ptr<SingleChainMCMC> const& coarseChain) {
  pt::ptree ptProposal;
  ptProposal.put("BlockIndex", 0);
  // subsampling is due to hyperparameter tuning
  const int subsampling = 5;
  ptProposal.put("Subsampling", subsampling);
  return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
}

std::shared_ptr<UQ::AbstractSamplingProblem>
UQ::MyParallelMIComponentFactory::SamplingProblem(std::shared_ptr<MultiIndex> const& index) {
  return std::make_shared<MySamplingProblem>(communicator, index, estimator);
}

std::shared_ptr<UQ::MIInterpolation> UQ::MyParallelMIComponentFactory::Interpolation([
    [maybe_unused]] std::shared_ptr<MultiIndex> const& index) {
  return std::make_shared<MyInterpolation>();
}

Eigen::VectorXd UQ::MyParallelMIComponentFactory::StartingPoint([
    [maybe_unused]] std::shared_ptr<MultiIndex> const& index) {
  Eigen::VectorXd start = Eigen::VectorXd::Ones(NUM_PARAM);
  // initial values
  const double initial_value = 0.5;
  start(0) = initial_value;
  start(1) = initial_value;
  return start;
}

UQ::MyParallelMIComponentFactory::MyParallelMIComponentFactory(const std::string& filename,
                                               std::shared_ptr<parcer::Communicator> communicator)
    : estimator(ODEModel::LikelihoodEstimator(filename)), communicator(std::move(communicator)) {}

void UQ::MyParallelMIComponentFactory::SetComm(const std::shared_ptr<parcer::Communicator>& comm) {
  communicator = comm;
}
