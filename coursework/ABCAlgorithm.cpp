#include "ABCAlgorithm.h"
#include "Utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

ABCAlgorithm::ABCAlgorithm(const JobShopInstance& inst, int sn, int lim, int t, int simulationRuns)
    : instance(inst), SN(sn), limit(lim), T(t), maxIterations(0), simulationRuns(simulationRuns) {
    population.reserve(SN);
    trials.resize(SN, 0);
    fitness.resize(SN, 0.0);
}

double ABCAlgorithm::calculateFitness(const Schedule& s) const {
    return 1.0 / (1.0 + s.cachedLmaxMean);
}

void ABCAlgorithm::updateFitnessValues() {
    for (int i = 0; i < SN; ++i) {
        fitness[i] = calculateFitness(population[i]);
    }
}

int ABCAlgorithm::rouletteWheelSelection() const {
    double totalFitness = std::accumulate(fitness.begin(), fitness.end(), 0.0);
    double randVal = randDouble(0.0, std::max(0.0, totalFitness));
    double sum = 0.0;
    for (int i = 0; i < SN; ++i) {
        sum += fitness[i];
        if (sum >= randVal) 
           return i;
    }
    return randInt(0, SN - 1);
}

bool ABCAlgorithm::isSignificantlyBetter(const Schedule& a, const Schedule& b, double alpha) {
    if (a.evalCount < 2 || b.evalCount < 2) {
        return a.cachedLmaxMean < b.cachedLmaxMean;
    }

    double mean1 = a.cachedLmaxMean;
    double mean2 = b.cachedLmaxMean;
    double var1 = a.getVariance();
    double var2 = b.getVariance();
    int n1 = a.evalCount;
    int n2 = b.evalCount;

    double se = std::sqrt(var1 / n1 + var2 / n2);
    if (se < 1e-9) return mean1 < mean2;

    double z = (mean1 - mean2) / se;
    double criticalValue = 1.96;

    return z < -criticalValue;
}

void ABCAlgorithm::allocateBudget(std::vector<Schedule>& solutions, int totalBudget) {
    size_t K = solutions.size();
    if (K == 0) return;

    int delta = std::max(10, totalBudget / simulationRuns);
    std::vector<int> counts(K, 0);

    for (int i = 0; i < K; ++i) {
        if (solutions[i].evalCount < requiredSimulations) {
            solutions[i].evaluateMC(requiredSimulations - solutions[i].evalCount);
        }
        counts[i] = solutions[i].evalCount;
    }

    int totalUsed = std::accumulate(counts.begin(), counts.end(), 0);

    while (totalUsed < totalBudget) {
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
        solutions[bestIdx].evaluateMC(delta);
        counts[bestIdx] += delta;
        totalUsed += delta;
    }
}

void ABCAlgorithm::preScreenAndEvaluate(Schedule& newSol, const Schedule& currentSol) {
    auto lowerBound = newSol.evaluateLowerBound();
    newSol.cachedLmaxLBJob = lowerBound.first;
    newSol.cachedLmaxLB = lowerBound.second;

    if (newSol.cachedLmaxLB > currentSol.cachedLmaxMean) {
        return;
    }

    newSol.evaluateMC(simulationRuns);
}

void ABCAlgorithm::initialize() {
    population.clear();
    for (int i = 0; i < SN; ++i) {
        population.emplace_back(&instance);
        Schedule& s = population.back();
        s.evaluateLowerBound();
        s.evaluateMC(simulationRuns);
        trials[i] = 0;
    }
    updateFitnessValues();
}

void ABCAlgorithm::run(int maxIter) {
    maxIterations = maxIter;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Employed bees phase
        for (int i = 0; i < SN; ++i) {
            Schedule newSol = population[i].getNeighborByArcOrientation();
            preScreenAndEvaluate(newSol, population[i]);

            if (newSol.cachedLmaxMean <= population[i].cachedLmaxMean) {
                population[i] = newSol;
                trials[i] = 0;
            } else {
                trials[i]++;
            }
        }

        updateFitnessValues();

        int onlookerCount = 0;
        while (onlookerCount < SN) {
            int selectedIdx = rouletteWheelSelection();
            Schedule newSol = population[selectedIdx].getNeighborByArcOrientation();
            preScreenAndEvaluate(newSol, population[selectedIdx]);

            if (newSol.cachedLmaxMean < population[selectedIdx].cachedLmaxMean) {
                population[selectedIdx] = newSol;
                trials[selectedIdx] = 0;
            } else {
                trials[selectedIdx]++;
            }
            onlookerCount++;
        }

        for (int i = 0; i < SN; ++i) {
            if (trials[i] >= limit) {
                // 'scout' bee generates random solution
                Schedule newSol(&instance);
                const int rndCount = randInt(0, instance.numJobs / 2);
                for (int i=0; i < rndCount; ++i)
                   newSol = newSol.getNeighborByArcOrientation();
                //Schedule newSol = getBestSolutionInPopulation();
                newSol.evaluateMC(simulationRuns);
                population[i] = newSol;
                trials[i] = 0;
            }
        }

        allocateBudget(population, T);
        updateFitnessValues();

        bestSolution = getBestSolutionInPopulation();

        /*if ((iter + 1) % 10 == 0)
        {
            std::cout << "Iteration " << (iter + 1) << "/" << maxIter
                << ", Best Lmax = " << bestSolution.cachedLmaxMean << std::endl;
        }*/
    }
}

const Schedule& ABCAlgorithm::getBestSolutionInPopulation() const {
   int bestIdx = 0;
   double bestValue = population[0].cachedLmaxMean;

   for (int i = 1; i < SN; ++i) {
      if (population[i].cachedLmaxMean < bestValue) {
         bestValue = population[i].cachedLmaxMean;
         bestIdx = i;
      }
   }

   return population[bestIdx];
}

const Schedule& ABCAlgorithm::getBestSolution() const
{
   return bestSolution;
}

double ABCAlgorithm::getBestFitness() const {
    const Schedule& best = getBestSolution();
    return calculateFitness(best);
}