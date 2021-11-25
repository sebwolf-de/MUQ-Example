#pragma once

#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelMIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

namespace UQ {
class MyStaticLoadBalancer : public muq::SamplingAlgorithms::StaticLoadBalancer {
  public:
  void
  virtual setup(std::shared_ptr<muq::SamplingAlgorithms::ParallelizableMIComponentFactory> componentFactory,
        uint availableRanks) override;
  virtual int numCollectors(std::shared_ptr<MultiIndex> modelIndex) override;
  virtual WorkerAssignment numWorkers(std::shared_ptr<MultiIndex> modelIndex) override;
  virtual ~MyStaticLoadBalancer(){};

  private:
  size_t ranks_remaining{};
  size_t models_remaining{};
};
} // namespace UQ
