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

#include "ODEModel/LikelihoodEstimator.h"
#include "ODEModel/ODESolver.h"

namespace uq {

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

class MySamplingProblem : public AbstractSamplingProblem {
  public:
  constexpr static double badLogDensity = -1200000;
  MySamplingProblem(const std::shared_ptr<MultiIndex>& index, std::shared_ptr<ode_model::ODESolver> runner,
                    size_t numberOfParameters, size_t numberOfFusedSims,
                    const std::string& referenceFileName,
                    std::shared_ptr<muq::Modeling::Gaussian> targetIn);

  double LogDensity(std::shared_ptr<SamplingState> const& state) override;
  std::shared_ptr<SamplingState> QOI() override;
  // Needed for MALAProposal:
  Eigen::VectorXd GradLogDensity(std::shared_ptr<SamplingState> const& state,
                                 unsigned blockWrt) override;

  private:
  std::shared_ptr<ode_model::ODESolver> runner;
  std::shared_ptr<SamplingState> lastState = nullptr;
  std::shared_ptr<MultiIndex> index;
  ode_model::LikelihoodEstimator likelihoodEstimator;

  size_t numberOfParameters;
  size_t numberOfFusedSims;

  static size_t runCount;

  /// The target distribution (the prior in the inference case)
  std::shared_ptr<muq::Modeling::Gaussian> target;
};

} // namespace uq
