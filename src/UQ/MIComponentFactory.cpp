#include "UQ/MIComponentFactory.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include "UQ/MIInterpolation.h"
#include "UQ/SamplingProblem.h"

#include "ODEModel/ODESolver.h"

#include "MUQ/SamplingAlgorithms/InfMALAProposal.h"
#include "MUQ/SamplingAlgorithms/MALAProposal.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"

std::shared_ptr<uq::MCMCProposal> uq::MyMIComponentFactory::Proposal(
    [[maybe_unused]] std::shared_ptr<MultiIndex> const& index,
    std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) {
  pt::ptree pt;
  pt.put("BlockIndex", 0);

  // TODO: make flexible
  const size_t numberOfParameters = 2;

  auto mu = Eigen::VectorXd::Zero(numberOfParameters);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(numberOfParameters, numberOfParameters);
  // Eigen::VectorXd cov = Eigen::VectorXd::Ones(numberOfParameters);

  for (size_t i = 0; i < numberOfParameters; i++) {
    cov(i, i) = startingParameters.variances(i); // cov(i, i)
  }

  auto prior = std::make_shared<Gaussian>(mu, cov, Gaussian::Mode::Covariance);

  return std::make_shared<MHProposal>(pt, samplingProblem, prior);
  // return std::make_shared<MALAProposal>(pt, samplingProblem, prior);
  // return std::make_shared<InfMALAProposal>(pt, samplingProblem, prior);
}

std::shared_ptr<uq::MultiIndex> uq::MyMIComponentFactory::FinestIndex() {
  auto index = std::make_shared<MultiIndex>(1);
  index->SetValue(0, finestIndex);
  return index;
}

std::shared_ptr<uq::MCMCProposal> uq::MyMIComponentFactory::CoarseProposal(
    [[maybe_unused]] std::shared_ptr<MultiIndex> const& fineIndex,
    [[maybe_unused]] std::shared_ptr<MultiIndex> const& coarseIndex,
    std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
    std::shared_ptr<SingleChainMCMC> const& coarseChain) {
  pt::ptree ptProposal;
  ptProposal.put("BlockIndex", 0);
  const int subsampling = 5;
  ptProposal.put("MLMCMC.Subsampling_0", subsampling);
  return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex,
                                                 coarseChain);
}

std::shared_ptr<uq::AbstractSamplingProblem>
uq::MyMIComponentFactory::SamplingProblem(std::shared_ptr<MultiIndex> const& index) {
  auto mu = Eigen::VectorXd::Zero(numberOfParameters);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(numberOfParameters, numberOfParameters);
  // Eigen::VectorXd cov = Eigen::VectorXd::Ones(numberOfParameters);

  for (size_t i = 0; i < numberOfParameters; i++) {
    cov(i, i) = startingParameters.variances(i); // cov(i, i)
  }

  auto prior = std::make_shared<Gaussian>(mu, cov, Gaussian::Mode::Covariance);
  return std::make_shared<MySamplingProblem>(index, runner, numberOfParameters, numberOfFusedSims,
                                             referenceFileName, prior);
}

std::shared_ptr<uq::MIInterpolation>
uq::MyMIComponentFactory::Interpolation([[maybe_unused]] std::shared_ptr<MultiIndex> const& index) {
  return std::make_shared<MyInterpolation>();
}

Eigen::VectorXd
uq::MyMIComponentFactory::StartingPoint([[maybe_unused]] std::shared_ptr<MultiIndex> const& index) {
  return startingParameters.values;
}

uq::MyMIComponentFactory::MyMIComponentFactory(std::shared_ptr<ode_model::ODESolver> runner,
                                               const uq::ValuesAndVariances& startingParameters,
                                               size_t finestIndex, size_t numberOfParameters,
                                               size_t numberOfFusedSims,
                                               std::string referenceFileName)
    : runner(std::move(runner)), startingParameters(std::move(startingParameters)),
      finestIndex(finestIndex), numberOfParameters(numberOfParameters),
      numberOfFusedSims(numberOfFusedSims), referenceFileName(std::move(referenceFileName)) {}
