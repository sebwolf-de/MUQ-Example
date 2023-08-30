#include "UQ/MIInterpolation.h"

std::shared_ptr<uq::SamplingState> uq::MyInterpolation::Interpolate(
    std::shared_ptr<SamplingState> const& coarseProposal,
    [[maybe_unused]] std::shared_ptr<SamplingState> const& fineProposal) {
  return std::make_shared<SamplingState>(coarseProposal->state);
}
