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

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <sstream>

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Run Generalized Metropolis Hastings MCMC sampling" << std::endl;
    std::cout << "  Usage: ./gmh numberOfFusedSimulations numberOfSamples" << std::endl;
    std::cout << "  if numberOfFusedSims == 1, default to standard MH sampling." << std::endl;
    return 1;
  }
  const size_t numberOfFusedSims = std::atoi(argv[1]);
  const size_t numberOfSamples = std::atoi(argv[2]);

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
      runner, initialParameterValuesAndVariance, 1, numberOfParameters, numberOfFusedSims,
      "../data/true_solution.dat");

  auto index = std::make_shared<MultiIndex>(1);
  index->SetValue(0, 0);
  auto problem = miComponentFactory->SamplingProblem(index);
  auto proposal = miComponentFactory->Proposal(index, problem);

  // parameters for the sampler
  boost::property_tree::ptree pt;
  pt.put("verbosity", 1); // show some output
  pt.put("BurnIn", 32 / numberOfFusedSims);
  pt.put("NumSamples", numberOfSamples);
  pt.put("PrintLevel", 1);

  const unsigned int numberOfProposals = numberOfFusedSims;
  const unsigned int numberOfAcceptedProposals = numberOfFusedSims;
  pt.put("NumProposals", numberOfProposals);
  pt.put("NumAccepted", numberOfAcceptedProposals);

  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
  if (numberOfFusedSims > 1) {
    kernels[0] = std::make_shared<FusedGMHKernel>(pt, problem, proposal);
  } else {
    kernels[0] = std::make_shared<MHKernel>(pt, problem, proposal);
  }

  auto chain = std::make_shared<SingleChainMCMC>(pt, kernels);
  chain->SetState(initialParameterValuesAndVariance.values);
  const std::shared_ptr<SampleCollection> samps = chain->Run();

  std::stringstream filenameStream;
  filenameStream << "test" << numberOfFusedSims << ".h5";
  const auto filename = filenameStream.str();
  std::cout << filename << std::endl;
  samps->WriteToFile(filename);
  std::cout << "Sample Mean = " << samps->Mean().transpose() << std::endl;
  std::cout << "Variance = " << samps->Variance().transpose() << std::endl;
  std::cout << "ESS = " << samps->ESS().transpose() << std::endl;
  std::cout << "Finished all" << std::endl;

  return 0;
}
