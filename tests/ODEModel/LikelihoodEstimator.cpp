#include "ODEModel/Function.h"
#include "doctest.h"

#include "ODEModel/LikelihoodEstimator.h"
#include <sched.h>
#include <sys/cdefs.h>

TEST_CASE("Test the file reader") {
  const auto f = ode_model::readFromFile("../tests/ODEModel/true_solution.dat");
  CHECK(f.time(0) == doctest::Approx(0.0));
  CHECK(f.time(1) == doctest::Approx(0.1));
  CHECK(f.time(2) == doctest::Approx(0.2));
  CHECK(f.values(0) == doctest::Approx(0.0));
  CHECK(f.values(1) == doctest::Approx(-1.3));
  CHECK(f.values(2) == doctest::Approx(20));
}

TEST_CASE("Test the interpolator") {
  const Eigen::Vector3d t = {0.0, 0.1, 0.2};
  const Eigen::Vector3d x = {0.0, 1.0, -2.0};
  const ode_model::Function f = {t, x};
  const Eigen::Vector3d s = {0.05, 0.1, 0.11};
  const auto g = ode_model::interpolate(f, s);

  CHECK(g(0) == doctest::Approx(0.5));
  CHECK(g(1) == doctest::Approx(1.0));
  CHECK(g(2) == doctest::Approx(1-0.3));
}

TEST_CASE("Test log likelihood") {
  const Eigen::Vector3d t = {0.0, 0.5, 1.0};
  const Eigen::Vector3d x = {0.0, 1.0, -2.0};
  const ode_model::Function f = {t, x};
  const ode_model::LikelihoodEstimator le(f);

  CHECK(le.calculateLogLikelihood(f) == doctest::Approx(0.0));
  const Eigen::Matrix<double, 5, 1> t_ = {0, 0.25, 0.5, 0.75, 1.0};
  const Eigen::Matrix<double, 5, 1> y = {0, 0, 0, 0, 0};
  const ode_model::Function g = {t_, y};
  CHECK(le.calculateLogLikelihood(g) == doctest::Approx(-16.0));
}
