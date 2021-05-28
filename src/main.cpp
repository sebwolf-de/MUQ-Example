#include <boost/property_tree/ptree.hpp>
#include <mpi.h>

#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/SLMCMC.h"

#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

#include "UQ/MIComponentFactory.h"
#include "UQ/MIInterpolation.h"
#include "UQ/SamplingProblem.h"
#include "UQ/StaticLoadBalancer.h"
#include "spdlog/common.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  spdlog::set_level(spdlog::level::debug);

  boost::property_tree::ptree pt;
  const size_t N = 1e2;
  pt.put("verbosity", 1); // show some output
  pt.put("MCMC.BurnIn", 1);
  pt.put("NumSamples_0", 100 * N);
  pt.put("NumSamples_1", N);
  pt.put("MLMCMC.Scheduling", true);
  pt.put("MLMCMC.Subsampling", 0);
  pt.put("MLMCMC.Subsampling_0", 0);
  pt.put("MLMCMC.Subsampling_1", 0);

  auto comm = std::make_shared<parcer::Communicator>(MPI_COMM_WORLD);
  auto localFactory = std::make_shared<UQ::MyMIComponentFactory>("true_solution.dat", comm);
  muq::SamplingAlgorithms::StaticLoadBalancingMIMCMC mimcmc(
      pt, localFactory, std::make_shared<UQ::MyStaticLoadBalancer>(), comm);

  if (comm->GetRank() == 0) {
    mimcmc.Run();
    Eigen::VectorXd meanQOI = mimcmc.MeanQOI();
    spdlog::info("mean QOI: ({}, {})", meanQOI(0), meanQOI(1));
    mimcmc.WriteToFile("samples.h5");
  }

  mimcmc.Finalize();
  MPI_Finalize();

  return 0;
}
