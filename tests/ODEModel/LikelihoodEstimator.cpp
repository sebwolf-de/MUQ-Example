#include "doctest.h"

#include "ODEModel/LikelihoodEstimator.h"
#include <sched.h>
#include <sys/cdefs.h>

TEST_CASE("Test the file reader") {
  const auto f = ODEModel::readFromFile("../tests/ODEModel/true_solution.dat");
  CHECK(f.time(0) == doctest::Approx(0.0));
  CHECK(f.time(1) == doctest::Approx(0.1));
  CHECK(f.time(2) == doctest::Approx(0.2));
  CHECK(f.x(0) == doctest::Approx(0.0));
  CHECK(f.x(1) == doctest::Approx(-1.3));
  CHECK(f.x(2) == doctest::Approx(20));
}

TEST_CASE("Test the interpolator") {
  const Eigen::Vector3d t = {0.0, 0.1, 0.2};
  const Eigen::Vector3d x = {0.0, 1.0, -2.0};
  const ODEModel::function f = {t, x};
  const Eigen::Vector3d s = {0.05, 0.1, 0.11};
  const auto g = ODEModel::interpolate(f, s);

  CHECK(g(0) == doctest::Approx(0.5));
  CHECK(g(1) == doctest::Approx(1.0));
  CHECK(g(2) == doctest::Approx(1-0.3));
}

TEST_CASE("Test log likelihood") {
  const Eigen::Vector3d t = {0.0, 0.5, 1.0};
  const Eigen::Vector3d x = {0.0, 1.0, -2.0};
  const ODEModel::function f = {t, x};
  const ODEModel::LikelihoodEstimator le(f);

  CHECK(le.caluculateLogLikelihood(x) == doctest::Approx(0.0));
  const Eigen::Matrix<double, 5, 1> y = Eigen::Matrix<double, 5, 1>::Zero();
  CHECK(le.caluculateLogLikelihood(y) == doctest::Approx(-4.0));
}
