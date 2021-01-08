#include "ODEModel.h"

void ODEModel::MyODEPiece::EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) {
  Eigen::Vector2d u_0 = {inputs.at(0)(0), 0.0}; // extract initial value from input
  const double omega = inputs.at(0)(1);         // extract frequency

  Eigen::Matrix<double, 2, Eigen::Dynamic> u(2, N);

  // solve IVP
  ODESolver::ImplicitEuler solver(omega);
  solver.solveIVP(u_0, u, dt);

  // extract interesting values
  outputs.resize(1);
  outputs.at(0) = u.block<1, N>(0, 0);
}
