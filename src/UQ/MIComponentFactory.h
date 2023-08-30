#pragma once

#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/MIInterpolation.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
#include "MUQ/Utilities/MultiIndices/MultiIndex.h"

#include "ODEModel/ODESolver.h"
#include "ValuesAndVariances.h"

namespace uq {

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

class MyMIComponentFactory : public MIComponentFactory {
  public:
  std::shared_ptr<MCMCProposal>
  Proposal(std::shared_ptr<MultiIndex> const& index,
           std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override;
  std::shared_ptr<MultiIndex> FinestIndex() override;
  std::shared_ptr<MCMCProposal>
  CoarseProposal(std::shared_ptr<MultiIndex> const& fineIndex,
                 std::shared_ptr<MultiIndex> const& coarseIndex,
                 std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                 std::shared_ptr<SingleChainMCMC> const& coarseChain) override;
  std::shared_ptr<AbstractSamplingProblem>
  SamplingProblem(std::shared_ptr<MultiIndex> const& index) override;
  std::shared_ptr<MIInterpolation> Interpolation(std::shared_ptr<MultiIndex> const& index) override;
  Eigen::VectorXd StartingPoint(std::shared_ptr<MultiIndex> const& index) override;

  MyMIComponentFactory(std::shared_ptr<ode_model::ODESolver> runner,
                       const uq::ValuesAndVariances& startingParameters, size_t finestIndex,
                       size_t numberOfParameters, size_t numberOfFusedSims,
                       std::string referenceFileName);

  private:
  const std::shared_ptr<ode_model::ODESolver> runner;
  const uq::ValuesAndVariances& startingParameters;
  const size_t finestIndex;
  size_t numberOfParameters;
  size_t numberOfFusedSims;
  std::string referenceFileName;
};

} // namespace uq
