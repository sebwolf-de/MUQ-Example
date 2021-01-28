#pragma once

#include <Eigen/Dense>

namespace ODEModel {
  class LikelihoodEstimator {
    private:
      struct function {
        Eigen::VectorXd time;
        Eigen::VectorXd x;
      };

      const function reference_from_file;

      function readFromFile(const std::string& file) const;
      Eigen::VectorXd interpolate(const function& f, const Eigen::VectorXd& other_time) const;

    public: 
      LikelihoodEstimator(const std::string& file);
      double caluculateLogLikelihood(const Eigen::Matrix<double, 1, Eigen::Dynamic>& solution) const;
  };
} // namespace ODEModel
