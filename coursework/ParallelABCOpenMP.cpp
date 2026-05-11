#include "ParallelABCOpenMP.h"
#include "Utils.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

ParallelABCOpenMP::ParallelABCOpenMP(const JobShopInstance& inst, int sn, int lim, int t, int simulationRuns, int threads)
    : ABCAlgorithm(inst, sn, lim, t, simulationRuns), numThreads(threads) {
}

void ParallelABCOpenMP::updateFitnessValuesParallel() {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int i = 0; i < SN; ++i) {
        fitness[i] = 1.0 / (1.0 + population[i].cachedLmaxMean);
    }
}

void ParallelABCOpenMP::parallelEmployedPhase() {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int i = 0; i < SN; ++i) {
        Schedule newSol = population[i].getNeighborByArcOrientation();
        
        auto lowerBound = newSol.evaluateLowerBound();
        newSol.cachedLmaxLBJob = lowerBound.first;
        newSol.cachedLmaxLB = lowerBound.second;

        if (newSol.cachedLmaxLB > population[i].cachedLmaxMean) {
            #pragma omp atomic
            trials[i]++;
            continue;
        }
        
        newSol.evaluateMCParallel(simulationRuns, numThreads);
        
        if (newSol.cachedLmaxMean <= population[i].cachedLmaxMean) {
            #pragma omp critical
            {
                if (newSol.cachedLmaxMean <= population[i].cachedLmaxMean) {
                    population[i] = newSol;
                    trials[i] = 0;
                }
            }
        } else {
            #pragma omp atomic
            trials[i]++;
        }
    }
}

void ParallelABCOpenMP::parallelOnlookerPhase() {
    updateFitnessValuesParallel();
    
    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int j = 0; j < SN; ++j) {
        int selectedIdx = rouletteWheelSelection();
        Schedule newSol = population[selectedIdx].getNeighborByArcOrientation();
        
        { // pre-screening
           auto lowerBound = newSol.evaluateLowerBound();
           newSol.cachedLmaxLBJob = lowerBound.first;
           newSol.cachedLmaxLB = lowerBound.second;

           if (lowerBound.second > population[selectedIdx].cachedLmaxMean) {
              #pragma omp atomic
              trials[selectedIdx]++;
              continue;
           }

           newSol.evaluateMCParallel(simulationRuns, numThreads);
        }

        if (newSol.cachedLmaxMean < population[selectedIdx].cachedLmaxMean) {
            #pragma omp critical
            {
                population[selectedIdx] = newSol;
                trials[selectedIdx] = 0;
            }
        } else {
            #pragma omp atomic
            trials[selectedIdx]++;
        }
    }
}

void ParallelABCOpenMP::parallelBudgetAllocation(std::vector<Schedule>& solutions, int totalBudget) {
    int K = SN;
    int delta = std::max(10, T / simulationRuns);
    
    std::vector<int> counts(K, 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int i = 0; i < K; ++i) {
        if (solutions[i].evalCount < requiredSimulations) {
            const_cast<Schedule&>(solutions[i]).evaluateMC(requiredSimulations - solutions[i].evalCount);
        }
        counts[i] = solutions[i].evalCount;
    }
    
    int totalUsed = std::accumulate(counts.begin(), counts.end(), 0);
    
    while (totalUsed < T) {
        std::vector<double> ucb(K);
        for (size_t i = 0; i < K; ++i) {
            if (counts[i] == 0) {
                ucb[i] = 1e9;
                continue;
            }
            double mean = solutions[i].cachedLmaxMean;
            double exploration = std::sqrt(2.0 * std::log(totalUsed + 1) / counts[i]);
            ucb[i] = mean - exploration;
        }
        
        auto bestIdx = std::distance(ucb.begin(), std::min_element(ucb.begin(), ucb.end()));
        solutions[bestIdx].evaluateMCParallel(delta, numThreads);
        counts[bestIdx] += delta;
        totalUsed += delta;
    }
}

void ParallelABCOpenMP::initialize() {
    population.resize(SN, Schedule(&instance));
    
    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int i = 0; i < SN; ++i) {
        Schedule& s = population[i];
        s.evaluateLowerBound();
        s.evaluateMCParallel(simulationRuns, numThreads);
        trials[i] = 0;
    }
    
    updateFitnessValuesParallel();
}

void ParallelABCOpenMP::run(int maxIter) {
    maxIterations = maxIter;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        parallelEmployedPhase();
        parallelOnlookerPhase();

        #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
        for (int i = 0; i < SN; ++i) {
            if (trials[i] >= limit) {
                // 'scout' bee generates random solution
                Schedule newSol(&instance);
                const int rndCount = randInt(0, instance.numJobs / 2);
                for (int i = 0; i < rndCount; ++i)
                    newSol = newSol.getNeighborByArcOrientation();
                //Schedule newSol = getBestSolutionInPopulation();
                newSol.evaluateMCParallel(simulationRuns, numThreads);
                population[i] = newSol;
                trials[i] = 0;
            }
        }

        parallelBudgetAllocation(population, T);
        updateFitnessValuesParallel();

        bestSolution = getBestSolutionInPopulation();
        
        /*if ((iter + 1) % 10 == 0)
        {
            std::cout << "Iteration " << (iter + 1) << "/" << maxIter
                      << ", Best Lmax = " << bestSolution.cachedLmaxMean << std::endl;
        }*/
    }
}
