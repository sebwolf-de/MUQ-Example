#ifndef UQ_VALUESANDVARIANCES_H
#define UQ_VALUESANDVARIANCES_H

#include <Eigen/Dense>

namespace uq {

struct ValuesAndVariances {
  Eigen::VectorXd values;
  Eigen::VectorXd variances;
};

} // namespace uq

#endif
