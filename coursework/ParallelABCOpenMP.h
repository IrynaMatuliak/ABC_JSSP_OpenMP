#ifndef PARALLEL_ABC_OPENMP_H
#define PARALLEL_ABC_OPENMP_H

#include "ABCAlgorithm.h"
#include <vector>
#include <omp.h>

class ParallelABCOpenMP : public ABCAlgorithm {
private:
    int numThreads;

    void parallelEmployedPhase();
    void parallelOnlookerPhase();
    void parallelBudgetAllocation(std::vector<Schedule>& solutions, int totalBudget);
    void updateFitnessValuesParallel();

public:
    ParallelABCOpenMP(const JobShopInstance& inst, int sn, int lim, int t, int simulationRuns, int threads);
    ~ParallelABCOpenMP() = default;

    void initialize() override;
    void run(int maxIter) override;
};

#endif