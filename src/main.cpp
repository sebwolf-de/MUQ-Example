#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelMIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include <boost/property_tree/ptree.hpp>

#include "ComponentFactory.h"
#include "Interpolation.h"
#include "SamplingProblem.h"

int main(int argc, char** argv) {

  auto localFactory = std::make_shared<UQ::MyMIComponentFactory>();

  boost::property_tree::ptree pt;
  pt.put("NumSamples", 1e3); // number of samples for single level
  pt.put("verbosity", 1);    // show some output
  pt.put("BurnIn", 10);

  muq::SamplingAlgorithms::SLMCMC slmcmc(pt, localFactory);
  std::shared_ptr<muq::SamplingAlgorithms::SampleCollection> samples = slmcmc.Run();

  std::cout << "SL mean Param: " << slmcmc.MeanParameter().transpose() << std::endl;
  std::cout << "SL mean QOI: " << slmcmc.MeanQOI().transpose() << std::endl;
  samples->WriteToFile("test.hdf5");

  return 0;
}
