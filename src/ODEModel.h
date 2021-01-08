#pragma once

#include <MUQ/Modeling/ModPiece.h>

#include "ODESolver.h"

namespace ODEModel {

class MyODEPiece : public muq::Modeling::ModPiece {
  public:
  MyODEPiece()
      : muq::Modeling::ModPiece(2 * Eigen::VectorXi::Ones(1),    // inputSizes = [2]
                                N * Eigen::VectorXi::Ones(1)){}; // outputSizes = [100]
  protected:
  // sample 1 second
  static constexpr size_t N = 1001;
  static constexpr double dt = 0.001;

  // where the magic happens
  virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override;
};
} // namespace ODEModel
