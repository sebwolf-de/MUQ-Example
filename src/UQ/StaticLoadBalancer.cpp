#include "StaticLoadBalancer.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"
#include "MUQ/SamplingAlgorithms/ParallelMIMCMCWorker.h"

void UQ::MyStaticLoadBalancer::setup(
    std::shared_ptr<muq::SamplingAlgorithms::ParallelizableMIComponentFactory> componentFactory,
    uint availableRanks) {
  ranks_remaining = availableRanks;
  spdlog::info("Hi, from the load balancer :D Balancing load across {} ranks", availableRanks);
  auto indices = MultiIndexFactory::CreateFullTensor(componentFactory->FinestIndex()->GetVector());
  models_remaining = indices->Size();
}

int UQ::MyStaticLoadBalancer::numCollectors(
    [[maybe_unused]] std::shared_ptr<MultiIndex> modelIndex) {
  ranks_remaining--;
  return 1;
}

muq::SamplingAlgorithms::StaticLoadBalancer::WorkerAssignment
UQ::MyStaticLoadBalancer::numWorkers(std::shared_ptr<MultiIndex> modelIndex) {
  WorkerAssignment assignment{};
  assignment.numWorkersPerGroup = 1;
  assignment.numGroups = (ranks_remaining / models_remaining) / assignment.numWorkersPerGroup;

  spdlog::info("Of {}, assigning {} to model {}", ranks_remaining,
               assignment.numGroups * assignment.numWorkersPerGroup, *modelIndex);

  assert(assignment.numGroups * assignment.numWorkersPerGroup > 0);

  models_remaining--;
  ranks_remaining -= assignment.numGroups * assignment.numWorkersPerGroup;

  return assignment;
}
