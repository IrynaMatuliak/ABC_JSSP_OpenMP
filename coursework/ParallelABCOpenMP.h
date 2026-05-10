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
    ParallelABCOpenMP(JobShopInstance* inst, int sn, int lim, int t, int simulationRuns, int threads);
    ~ParallelABCOpenMP() = default;

    void initialize() override;
    void run(int maxIter) override;
    void setNumThreads(int threads) { numThreads = threads; omp_set_num_threads(threads); }
};

#endif