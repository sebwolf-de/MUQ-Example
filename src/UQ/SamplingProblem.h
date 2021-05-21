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
#include "ODEModel/ODEPiece.h"

static constexpr int NUM_PARAM = 2;

namespace UQ {

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

class MySamplingProblem : public AbstractSamplingProblem {
  public:
  MySamplingProblem(std::shared_ptr<parcer::Communicator> comm,
                    std::shared_ptr<parcer::Communicator> globalComm,
                    std::shared_ptr<MultiIndex> index,
                    const ODEModel::LikelihoodEstimator& likelihoodEstimator);

  virtual ~MySamplingProblem(){};

  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override;
  virtual std::shared_ptr<SamplingState> QOI() override;

  private:
  const ODEModel::LikelihoodEstimator& estimator;
  std::shared_ptr<parcer::Communicator> comm;
  std::shared_ptr<parcer::Communicator> globalComm;
  std::shared_ptr<MultiIndex> index;
  std::shared_ptr<SamplingState> lastState = nullptr;
};

} // namespace UQ
