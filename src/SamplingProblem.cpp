#include "SamplingProblem.h"

UQ::MySamplingProblem::MySamplingProblem(std::shared_ptr<MultiIndex> index)
    : AbstractSamplingProblem(Eigen::VectorXi::Constant(1, NUM_PARAM),
                              Eigen::VectorXi::Constant(1, NUM_PARAM)) {
  this->index = index;
  std::cout << "Run Sampling Problem with index" << index->GetValue(0) << std::endl;
}

double UQ::MySamplingProblem::LogDensity(std::shared_ptr<SamplingState> const& state) {
  lastState = state;

  // Discard stupid parameters
  if (state->state[0][0] > 1.0 || state->state[0][0] < 0.0 || state->state[0][1] > 1.0 ||
      state->state[0][1] < 0.0)
    return -24;

  const size_t N = std::pow(35, (index->GetValue(0)+1)) + 1;
  auto piece = std::make_shared<ODEModel::MyODEPiece>(N);
  std::vector<Eigen::VectorXd> inputs(1);
  inputs.at(0) = state->state[0];
  std::vector<Eigen::VectorXd> outputs = piece->Evaluate(inputs);
  Eigen::Matrix<double, 1, Eigen::Dynamic> solution = outputs.at(0);
  const auto likelihood = ODEModel::caluculateLikelihood(solution);

  // Create some debug output
  std::cout << "DOFs: " << N << ", parameter:" << state->state[0].transpose() << ", likelihood: " << likelihood
            << std::endl;
  return likelihood;
}

std::shared_ptr<UQ::SamplingState> UQ::MySamplingProblem::QOI() {
  assert(lastState != nullptr);
  return std::make_shared<SamplingState>(lastState->state, 1.0);
}
