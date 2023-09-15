#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"

#include "MUQ/SamplingAlgorithms/FusedGMHKernel.h"
#include "MUQ/SamplingAlgorithms/SampleCollection.h"
#include "MUQ/SamplingAlgorithms/TransitionKernel.h"
#include "MUQ/Utilities/MultiIndices/MultiIndex.h"

#include <boost/property_tree/ptree.hpp>

#include "ODEModel/LikelihoodEstimator.h"
#include "ODEModel/ODESolver.h"
#include "UQ/MIComponentFactory.h"
#include "UQ/MIInterpolation.h"
#include "UQ/SamplingProblem.h"

#include "spdlog/spdlog.h"

#include "argparse/argparse.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <sstream>

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

struct MyArgs : public argparse::Args {
  int& numberOfSamples = arg("numberOfSamples");
  int& numberOfFusedSims =
      kwarg("n,numberOfFusedSims", "Number of fused simulations").set_default(4);
  int& numberOfAcceptedProposals =
      kwarg("k,numberOfAcceptedProposals", "Number of accepted proposals").set_default(4);
};

int main(int argc, char* argv[]) {
  MyArgs args = argparse::parse<MyArgs>(argc, argv);
  args.print();                            // prints all variables
  spdlog::set_level(spdlog::level::debug); // Set global log level to debug

  Eigen::VectorXd initialValues(2);
  initialValues << 0.1, -0.1;
  Eigen::VectorXd variances(2);
  variances << 1.0, 1.0;
  auto initialParameterValuesAndVariance = uq::ValuesAndVariances{initialValues, variances};

  const size_t numberOfParameters = 2;
  const double omega = 1.0;
  const double dt = 0.1;
  const double numberOfTimesteps = 11;

  const auto runner = std::make_shared<ode_model::ImplicitEuler>(omega, dt, numberOfTimesteps);

  auto miComponentFactory = std::make_shared<uq::MyMIComponentFactory>(
      runner, initialParameterValuesAndVariance, 1, numberOfParameters, args.numberOfFusedSims,
      "../data/true_solution.dat");

  auto index = std::make_shared<MultiIndex>(1);
  index->SetValue(0, 0);
  auto problem = miComponentFactory->SamplingProblem(index);
  auto proposal = miComponentFactory->Proposal(index, problem);

  // parameters for the sampler
  boost::property_tree::ptree pt;
  pt.put("verbosity", 1); // show some output
  pt.put("BurnIn", 32 / args.numberOfFusedSims);
  pt.put("NumSamples", args.numberOfSamples);
  pt.put("PrintLevel", 1);

  const unsigned int numberOfProposals = args.numberOfFusedSims;
  const unsigned int numberOfAcceptedProposals = args.numberOfAcceptedProposals;
  pt.put("NumProposals", numberOfProposals);
  pt.put("NumAccepted", numberOfAcceptedProposals);

  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
  if (args.numberOfFusedSims > 1) {
    kernels[0] = std::make_shared<FusedGMHKernel>(pt, problem, proposal);
  } else {
    kernels[0] = std::make_shared<MHKernel>(pt, problem, proposal);
  }

  auto chain = std::make_shared<SingleChainMCMC>(pt, kernels);
  chain->SetState(initialParameterValuesAndVariance.values);
  const std::shared_ptr<SampleCollection> samps = chain->Run();

  const Eigen::MatrixXd sampleMatrix = samps->AsMatrix();
  const unsigned int matrixColumns = sampleMatrix.cols();
  const Eigen::MatrixXd firstSamples = sampleMatrix(0, Eigen::seq(0, matrixColumns - 2));
  const Eigen::MatrixXd lastSamples = sampleMatrix(0, Eigen::lastN(matrixColumns - 1));
  const Eigen::ArrayXXd absoluteDiff = (lastSamples - firstSamples).array().abs();
  const Eigen::ArrayXX<bool> nonZeroDiff = absoluteDiff > 0;
  const unsigned changes = absoluteDiff.count();

  std::stringstream filenameStream;
  filenameStream << "test-" << args.numberOfFusedSims << "-" << args.numberOfAcceptedProposals
                 << ".h5";
  const auto filename = filenameStream.str();
  samps->WriteToFile(filename);
  std::cout << "Sample Mean = " << samps->Mean().transpose() << std::endl;
  std::cout << "Executed solver " << ode_model::ImplicitEuler::numberOfExecutions
            << " times, to evaluate "
            << ode_model::ImplicitEuler::numberOfExecutions * args.numberOfFusedSims
            << " forward models." << std::endl;
  std::cout << "Variance = " << samps->Variance().transpose() << std::endl;
  std::cout << "ESS = " << samps->ESS().transpose() << std::endl;
  std::cout << "Acceptance Ratio " << changes << "/" << matrixColumns << " = "
            << static_cast<double>(changes) / static_cast<double>(matrixColumns) << std::endl;
  std::cout << "Finished all" << std::endl;

  return 0;
}
