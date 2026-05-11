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

JobShopInstance copyInstance(const JobShopInstance& original) {
    JobShopInstance copy(original.numJobs, original.numMachines, original.distType, original.theta);
    copy.jobs = original.jobs;
    copy.dueDates = original.dueDates;
    copy.jobWeights = original.jobWeights;
    copy.operationsPerMachine = original.operationsPerMachine;
    return copy;
}

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

struct SResult {
    int numThreads = 0;
    double avgTime = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double avgLmax = 0.0;
    double avgMakespan = 0.0;
    bool allFeasible = true;
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "Job Shop Scheduling\n";
    std::cout << "============================================================\n";

    const int numJobs = 1000;
    const int numMachines = 200;
    const int populationSize = 50;
    const int limit = 50;
    const int maxIterations = 1000;
    const double theta = 0.2;
    const int budget = 200;
    const auto distType = JobShopInstance::MEAN;
    const int simulationRuns = 5;
    const int numRuns = 5;

    std::cout << "\nConfiguration:\n";
    std::cout << "  Jobs: " << numJobs << "\n";
    std::cout << "  Machines: " << numMachines << "\n";
    std::cout << "  Iterations: " << maxIterations << "\n";
    std::cout << "  Population: " << populationSize << "\n";
    std::cout << "  Budget: " << budget << "\n";
    std::cout << "  NumRuns: " << numRuns << "\n";

    JobShopInstance baseInstance(numJobs, numMachines, distType, theta);
    if (!std::filesystem::exists("baseInstance.json")) {
        baseInstance.generateRandomInstance(42);
        baseInstance.writeToFile("baseInstance.json");
    } else {
        baseInstance.readFromFile("baseInstance.json");
        if (baseInstance.numJobs != numJobs || baseInstance.numMachines != numMachines) {
            baseInstance = JobShopInstance(numJobs, numMachines, distType, theta);
            baseInstance.generateRandomInstance(42);
            baseInstance.writeToFile("baseInstance.json");
        }
    }

    std::cout << "\nRunning SEQUENTIAL ABC (" << numRuns << " times)...\n";
    std::vector<double> sTimes, sLmaxs, sMakespans;
    bool allSeqFeasible = true;

    for (int i = 0; i < numRuns; ++i) {
        JobShopInstance inst = copyInstance(baseInstance);
        Timer t; t.start();
        ABCAlgorithm algo(inst, populationSize, limit, budget, simulationRuns);
        algo.initialize();
        algo.run(maxIterations);

        sTimes.push_back(t.elapsedSec());
        sLmaxs.push_back(algo.getBestSolution().cachedLmaxMean);
        sMakespans.push_back(calculateMakespan(algo.getBestSolution()));
        if (!quickCheckFeasibility(algo.getBestSolution())) allSeqFeasible = false;
    }

    double avgSeqTime = std::accumulate(sTimes.begin(), sTimes.end(), 0.0) / numRuns;
    double avgSeqLmax = std::accumulate(sLmaxs.begin(), sLmaxs.end(), 0.0) / numRuns;
    double avgSeqMakespan = std::accumulate(sMakespans.begin(), sMakespans.end(), 0.0) / numRuns;

    const std::vector<int> threadsCount = { 2, 4, 6, 7, 8, 10, 12, 16, 25 };
    std::vector<SResult> results;

    for (int nThreads : threadsCount) {
        std::cout << "Running PARALLEL ABC with " << nThreads << " threads (" << numRuns << " times)...\n";
        std::vector<double> pTimes, pLmaxs, pMakespans;
        bool allParFeasible = true;

        for (int i = 0; i < numRuns; ++i) {
            JobShopInstance inst = copyInstance(baseInstance);
            Timer t; t.start();
            ParallelABCOpenMP algo(inst, populationSize, limit, budget, simulationRuns, nThreads);
            algo.initialize();
            algo.run(maxIterations);

            pTimes.push_back(t.elapsedSec());
            pLmaxs.push_back(algo.getBestSolution().cachedLmaxMean);
            pMakespans.push_back(calculateMakespan(algo.getBestSolution()));
            if (!quickCheckFeasibility(algo.getBestSolution())) allParFeasible = false;
        }

        double avgTime = std::accumulate(pTimes.begin(), pTimes.end(), 0.0) / numRuns;
        SResult res;
        res.numThreads = nThreads;
        res.avgTime = avgTime;
        res.speedup = avgSeqTime / avgTime;
        res.efficiency = res.speedup / nThreads;
        res.avgLmax = std::accumulate(pLmaxs.begin(), pLmaxs.end(), 0.0) / numRuns;
        res.avgMakespan = std::accumulate(pMakespans.begin(), pMakespans.end(), 0.0) / numRuns;
        res.allFeasible = allParFeasible;
        results.push_back(res);
    }

    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << std::left << std::setw(12) << "Method"
        << std::setw(10) << "Threads"
        << std::setw(10) << "Time(s)"
        << std::setw(10) << "Speedup"
        << std::setw(12) << "Efficiency"
        << std::setw(12) << "Lmax"
        << std::setw(12) << "Makespan"
        << "Feasible\n";
    std::cout << std::string(90, '-') << "\n";

    std::cout << std::left << std::setw(12) << "Sequential"
        << std::setw(10) << "-"
        << std::fixed << std::setprecision(2) << std::setw(10) << avgSeqTime
        << std::setw(10) << "-"
        << std::setw(12) << "-"
        << std::setw(12) << avgSeqLmax
        << std::setw(12) << std::setprecision(1) << avgSeqMakespan
        << (allSeqFeasible ? "Yes" : "No") << "\n";

    for (const auto& res : results) {
        std::cout << std::left << std::setw(12) << "OpenMP"
            << std::setw(10) << res.numThreads
            << std::fixed << std::setprecision(2) << std::setw(10) << res.avgTime
            << std::setw(10) << res.speedup
            << std::setw(12) << res.efficiency
            << std::setw(12) << res.avgLmax
            << std::setw(12) << std::setprecision(1) << res.avgMakespan
            << (res.allFeasible ? "Yes" : "No") << "\n";
    }
    std::cout << std::string(90, '=') << "\n";

    return 0;
}