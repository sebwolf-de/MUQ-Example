#pragma once

#include <MUQ/Modeling/ModPiece.h>

#include "ODESolver.h"

namespace ODEModel {

class MyODEPiece : public muq::Modeling::ModPiece {
  public:
  MyODEPiece(size_t N)
      : muq::Modeling::ModPiece(2 * Eigen::VectorXi::Ones(1),  // inputSizes = [2]
                                N * Eigen::VectorXi::Ones(1)), // outputSizes = [N]
        N(N), dt(1 / double(N - 1)){};

  protected:
  // sample 1 second
  const size_t N;
  const double dt;

  // where the magic happens
  virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override;
};
} // namespace ODEModel
