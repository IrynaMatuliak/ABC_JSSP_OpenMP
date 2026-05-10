#include "JobShopInstance.h"
#include "ABCAlgorithm.h"
#include "ParallelABCOpenMP.h"
#include "Utils.h"
#include "AlgorithmValidator.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <filesystem>

std::mt19937 rng(12345);

double calculateMakespan(const Schedule& schedule) {
    int n = schedule.instance->numJobs;
    int m = schedule.instance->numMachines;
    std::vector<double> jobNextAvail(n, 0.0);
    std::vector<double> machNextAvail(m, 0.0);
    std::vector<int> jobStep(n, 0);

    std::vector<double> meanTimes(n * m);
    for (int j = 0; j < n; ++j)
        for (int s = 0; s < m; ++s)
            meanTimes[j * m + s] = schedule.instance->jobs[j][s].mean;

    for (int opIdx : schedule.operationSequence) {
        int job = opIdx / m;
        int step = jobStep[job];
        int machine = schedule.instance->jobs[job][step].machine;
        double duration = meanTimes[opIdx];
        double start = std::max(jobNextAvail[job], machNextAvail[machine]);
        double end = start + duration;

        jobNextAvail[job] = end;
        machNextAvail[machine] = end;
        jobStep[job]++;
    }

    double maxCompletion = 0;
    for (double c : jobNextAvail) maxCompletion = std::max(maxCompletion, c);
    return maxCompletion;
}

bool quickCheckFeasibility(const Schedule& schedule) {
    return schedule.buildOperationSequenceFromMachineSequences(nullptr);
}

bool fullFeasibilityCheck(const Schedule& schedule, bool verbose = false) {
    auto startTime = std::chrono::high_resolution_clock::now();

    bool result = schedule.verifyFullAcyclicity();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (verbose) {
        std::cout << "    Full feasibility check took " << duration.count() << " ms\n";
    }

    return result;
}

JobShopInstance copyInstance(const JobShopInstance& original) {
    JobShopInstance copy(original.numJobs, original.numMachines, original.distType, original.theta);
    copy.jobs = original.jobs;
    copy.dueDates = original.dueDates;
    copy.jobWeights = original.jobWeights;
    copy.operationsPerMachine = original.operationsPerMachine;
    return copy;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "Job Shop Scheduling\n";
    std::cout << "============================================================\n";

    const int numJobs = 1000;
    const int numMachines = 100;
    const int numThreads = 8;
    const int populationSize = 50;
    const int limit = 50;
    const int maxIterations = 1000;
    const double theta = 0.2;
    const int budget = 200;
    const auto distType = JobShopInstance::MEAN;
    const int numRuns = 10;

    // how many simulations run to test feasibility of the setup
    const int simulationRuns = 5;

    std::cout << "\nConfiguration:\n";
    std::cout << "  Jobs: " << numJobs << "\n";
    std::cout << "  Machines: " << numMachines << "\n";
    std::cout << "  Threads: " << numThreads << "\n";
    std::cout << "  Iterations: " << maxIterations << "\n";
    std::cout << "  Population: " << populationSize << "\n";
    std::cout << "  Budget: " << budget << "\n";
    std::cout << "  NumRuns: " << numRuns << "\n";

    JobShopInstance baseInstance(numJobs, numMachines, distType, theta);

    if (!std::filesystem::exists("baseInstance.json")) {
        baseInstance.generateRandomInstance(42);
        baseInstance.writeToFile("baseInstance.json");
        std::cout << "Instance generated.\n";
    } else {
        std::cout << "Instance read from file 'baseInstance.json'.\n";
        baseInstance.readFromFile("baseInstance.json");
        // if doesnt match - regenerate with new set of input parameters
        if (baseInstance.numJobs != numJobs || baseInstance.numMachines != numMachines || baseInstance.distType != distType || baseInstance.theta != theta) {
            baseInstance = JobShopInstance(numJobs, numMachines, distType, theta);
            baseInstance.generateRandomInstance(42);
            baseInstance.writeToFile("baseInstance.json");
            std::cout << "Saved data doesn`t match. Instance generated with new input parameters.\n";
        }
    }

    std::cout << "\nVerifying initial schedule feasibility...\n";
    Schedule testSchedule(&baseInstance);
    bool initialFeasible = testSchedule.isAcyclicFast();
    std::cout << "  Initial schedule is " << (initialFeasible ? "FEASIBLE" : "INFEASIBLE") << "\n";

    if (!initialFeasible) {
        std::cout << "  Repairing initial schedule...\n";
        testSchedule.repair();
        std::cout << "  After repair: " << (testSchedule.isAcyclicFast() ? "FEASIBLE" : "INFEASIBLE") << "\n";
    }


    std::vector<double> seqTimes(numRuns);
    std::vector<double> seqMakespans(numRuns);
    std::vector<double> seqLmaxValues(numRuns);
    std::vector<bool> seqFeasibleResults(numRuns);

    std::vector<double> ompTimes(numRuns);
    std::vector<double> ompMakespans(numRuns);
    std::vector<double> ompLmaxValues(numRuns);
    std::vector<bool> ompFeasibleResults(numRuns);

    int seqBetterCount = 0;
    int ompBetterCount = 0;
    int tieCount = 0;

    for (int run = 0; run < numRuns; ++run) {
        JobShopInstance seqInstance = copyInstance(baseInstance);
        Timer seqTimer;
        seqTimer.start();

        ABCAlgorithm sequential(&seqInstance, populationSize, limit, budget, simulationRuns);
        sequential.initialize();
        sequential.run(maxIterations);

        double seqTime = seqTimer.elapsedSec();
        const Schedule& bestSeq = sequential.getBestSolution();
        double seqMakespan = calculateMakespan(bestSeq);

        bool seqQuickFeasible = quickCheckFeasibility(bestSeq);
        bool seqFeasible = seqQuickFeasible;
        if (seqQuickFeasible) {
            seqFeasible = fullFeasibilityCheck(bestSeq, false);
        }

        seqTimes[run] = seqTime;
        seqMakespans[run] = seqMakespan;
        seqLmaxValues[run] = bestSeq.cachedLmaxMean;
        seqFeasibleResults[run] = seqFeasible;

        JobShopInstance ompInstance = copyInstance(baseInstance);
        Timer ompTimer;
        ompTimer.start();

        ParallelABCOpenMP ompParallel(&ompInstance, populationSize, limit, budget, simulationRuns, numThreads);
        ompParallel.initialize();
        ompParallel.run(maxIterations);

        double ompTime = ompTimer.elapsedSec();
        const Schedule& bestOmp = ompParallel.getBestSolution();
        double ompMakespan = calculateMakespan(bestOmp);

        bool ompQuickFeasible = quickCheckFeasibility(bestOmp);
        bool ompFeasible = ompQuickFeasible;
        if (ompQuickFeasible) {
            ompFeasible = fullFeasibilityCheck(bestOmp, false);
        }

        ompTimes[run] = ompTime;
        ompMakespans[run] = ompMakespan;
        ompLmaxValues[run] = bestOmp.cachedLmaxMean;
        ompFeasibleResults[run] = ompFeasible;

        if (bestOmp.cachedLmaxMean < bestSeq.cachedLmaxMean) {
            ompBetterCount++;
        } else if (bestSeq.cachedLmaxMean < bestOmp.cachedLmaxMean) {
            seqBetterCount++;
        } else {
            tieCount++;
        }
    }

    double avgSeqTime = std::accumulate(seqTimes.begin(), seqTimes.end(), 0.0) / numRuns;
    double avgSeqMakespan = std::accumulate(seqMakespans.begin(), seqMakespans.end(), 0.0) / numRuns;
    double avgSeqLmax = std::accumulate(seqLmaxValues.begin(), seqLmaxValues.end(), 0.0) / numRuns;

    double avgOmpTime = std::accumulate(ompTimes.begin(), ompTimes.end(), 0.0) / numRuns;
    double avgOmpMakespan = std::accumulate(ompMakespans.begin(), ompMakespans.end(), 0.0) / numRuns;
    double avgOmpLmax = std::accumulate(ompLmaxValues.begin(), ompLmaxValues.end(), 0.0) / numRuns;

    double avgSpeedup = avgSeqTime / avgOmpTime;
    double avgEfficiency = (avgSeqTime / avgOmpTime) / numThreads;

    bool finalSeqFeasible = std::all_of(seqFeasibleResults.begin(), seqFeasibleResults.end(), [](bool v) { return v; });
    bool finalOmpFeasible = std::all_of(ompFeasibleResults.begin(), ompFeasibleResults.end(), [](bool v) { return v; });

    std::cout << "\n============================================================\n";
    std::cout << "RESULTS\n";
    std::cout << "============================================================\n";
    std::cout << std::left << std::setw(15) << "Method"
        << std::setw(12) << "Time(s)"
        << std::setw(12) << "Speedup"
        << std::setw(12) << "Efficiency"
        << std::setw(15) << "Lmax"
        << std::setw(12) << "Makespan"
        << std::setw(12) << "Feasible" << "\n";
    std::cout << "------------------------------------------------------------\n";

    std::cout << std::left << std::setw(15) << "Sequential"
        << std::right << std::setw(10) << std::fixed << std::setprecision(2) << avgSeqTime
        << std::setw(12) << "-"
        << std::setw(12) << "-"
        << std::setw(15) << std::fixed << std::setprecision(2) << avgSeqLmax
        << std::setw(12) << std::fixed << std::setprecision(1) << avgSeqMakespan
        << std::setw(12) << (finalSeqFeasible ? "Yes" : "No") << "\n";

    std::cout << std::left << std::setw(15) << "OpenMP"
        << std::right << std::setw(10) << std::fixed << std::setprecision(2) << avgOmpTime
        << std::setw(12) << std::fixed << std::setprecision(2) << avgSpeedup
        << std::setw(12) << std::fixed << std::setprecision(2) << avgEfficiency
        << std::setw(15) << std::fixed << std::setprecision(2) << avgOmpLmax
        << std::setw(12) << std::fixed << std::setprecision(1) << avgOmpMakespan
        << std::setw(12) << (finalOmpFeasible ? "Yes" : "No") << "\n";

    std::cout << "\n============================================================\n";

    if (finalSeqFeasible && finalOmpFeasible) {
        std::cout << "Both schedules are FEASIBLE - No cycles detected\n";

        if (ompBetterCount > seqBetterCount) {
            std::cout << "Parallel algorithm found better solution\n";
        } else if (seqBetterCount > ompBetterCount) {
            std::cout << "Sequential algorithm found better solution\n";
        } else {
            if (avgOmpLmax < avgSeqLmax) {
                std::cout << "Parallel algorithm found better solution (better average Lmax)\n";
            } else if (avgSeqLmax < avgOmpLmax) {
                std::cout << "Sequential algorithm found better solution (better average Lmax)\n";
            } else {
                std::cout << "Both algorithms found similar quality solutions\n";
            }
        }

    } else if (!finalSeqFeasible && !finalOmpFeasible) {
        std::cout << "Both schedules are INFEASIBLE - Cycles detected\n";
    } else {
        std::cout << "Mixed results - One schedule is feasible, the other is not\n";
    }

    return 0;
}