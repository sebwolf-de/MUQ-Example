#pragma once

#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/MIInterpolation.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"
#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
#include "MUQ/Utilities/MultiIndices/MultiIndex.h"

#include "ODEModel/LikelihoodEstimator.h"
#include "parcer/Communicator.h"

namespace UQ {

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

class MyMIComponentFactory : public ParallelizableMIComponentFactory {
  public:
  virtual std::shared_ptr<MCMCProposal>
  Proposal(std::shared_ptr<MultiIndex> const& index,
           std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override;
  virtual std::shared_ptr<MultiIndex> FinestIndex() override;
  virtual std::shared_ptr<MCMCProposal>
  CoarseProposal(std::shared_ptr<MultiIndex> const& index,
                 std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                 std::shared_ptr<SingleChainMCMC> const& coarseChain) override;
  virtual std::shared_ptr<AbstractSamplingProblem>
  SamplingProblem(std::shared_ptr<MultiIndex> const& index) override;
  virtual std::shared_ptr<MIInterpolation>
  Interpolation(std::shared_ptr<MultiIndex> const& index) override;
  virtual Eigen::VectorXd StartingPoint(std::shared_ptr<MultiIndex> const& index) override;
  void SetComm(std::shared_ptr<parcer::Communicator> const& comm) override;

  MyMIComponentFactory(std::string filename,
                       std::shared_ptr<parcer::Communicator> globalCommunicator);

  private:
  const ODEModel::LikelihoodEstimator estimator;
  std::shared_ptr<parcer::Communicator> communicator;
  std::shared_ptr<parcer::Communicator> globalCommunicator;
};
} // namespace UQ
