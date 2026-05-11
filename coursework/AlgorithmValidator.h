#ifndef ALGORITHM_VALIDATOR_H
#define ALGORITHM_VALIDATOR_H

#include "ABCAlgorithm.h"
#include "ParallelABCOpenMP.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <limits>
#include <functional>
#include <vector>
#include <numeric>
#include <cmath>
#include <queue>
#include <algorithm>
#include <random>

class AlgorithmValidator {
private:
    struct ValidationResult {
        bool passed;
        std::string message;
        std::vector<std::string> warnings;

        ValidationResult() : passed(true) {}
    };

    ValidationResult checkAcyclicity(const Schedule& schedule) {
        ValidationResult result;
        if (!schedule.verifyFullAcyclicity()) {
            result.passed = false;
            result.message = "Schedule contains cycles";
            return result;
        }
        result.passed = true;
        result.message = "Schedule is acyclic";
        return result;
    }

    ValidationResult checkOperationCompleteness(const Schedule& schedule) {
        ValidationResult result;
        if (!schedule.instance) {
            result.passed = false;
            result.message = "No instance attached to schedule";
            return result;
        }

        int totalOps = schedule.instance->numJobs * schedule.instance->numMachines;

        if (schedule.operationSequence.size() != totalOps) {
            result.passed = false;
            result.message = "Missing operations. Expected: " +
                std::to_string(totalOps) + ", Got: " +
                std::to_string(schedule.operationSequence.size());
            return result;
        }

        std::set<int> uniqueOps(schedule.operationSequence.begin(),
            schedule.operationSequence.end());
        if (uniqueOps.size() != totalOps) {
            result.passed = false;
            result.message = "Duplicate or missing operations";
            return result;
        }

        result.passed = true;
        result.message = "All operations present";
        return result;
    }

    ValidationResult checkJobPrecedence(const Schedule& schedule) {
        ValidationResult result;
        std::vector<int> lastSeenStep(schedule.instance->numJobs, -1);

        for (int opId : schedule.operationSequence) {
            int job = opId / schedule.instance->numMachines;
            int step = opId % schedule.instance->numMachines;

            if (step != lastSeenStep[job] + 1) {
                result.passed = false;
                result.message = "Job " + std::to_string(job) +
                    " order violation. Expected step " +
                    std::to_string(lastSeenStep[job] + 1) +
                    ", got " + std::to_string(step);
                return result;
            }
            lastSeenStep[job] = step;
        }

        result.passed = true;
        result.message = "Job precedence satisfied";
        return result;
    }

    ValidationResult checkMachineSequences(const Schedule& schedule) {
        ValidationResult result;

        for (int m = 0; m < schedule.instance->numMachines; ++m) {
            const auto& seq = schedule.machineSequences[m];
            std::set<int> uniqueOps(seq.begin(), seq.end());

            if (uniqueOps.size() != seq.size()) {
                result.passed = false;
                result.message = "Duplicate operations on machine " + std::to_string(m);
                return result;
            }

            for (int opId : seq) {
                int job = opId / schedule.instance->numMachines;
                int step = opId % schedule.instance->numMachines;
                int machine = schedule.instance->jobs[job][step].machine;

                if (machine != m) {
                    result.passed = false;
                    result.message = "Operation assigned to wrong machine";
                    return result;
                }
            }
        }

        result.passed = true;
        result.message = "Machine sequences consistent";
        return result;
    }

public:
    struct FullValidationReport {
        bool acyclicity;
        bool completeness;
        bool precedence;
        bool machineSequences;
        bool overall;
        std::string details;

        void print(const std::string& title = "") const {
            if (!title.empty()) {
                std::cout << "  " << title << ":\n";
            }
            std::cout << "Acyclicity: " << (acyclicity ? "PASS" : "FAIL") << "\n";
            std::cout << "Completeness: " << (completeness ? "PASS" : "FAIL") << "\n";
            std::cout << "Precedence: " << (precedence ? "PASS" : "FAIL") << "\n";
            std::cout << "Machine Sequences: " << (machineSequences ? "PASS" : "FAIL") << "\n";
            std::cout << "Overall: " << (overall ? "PASSED" : "FAILED") << "\n";
            if (!details.empty()) {
                std::cout << "Details: " << details << "\n";
            }
        }
    };

    FullValidationReport validateSchedule(const Schedule& schedule) {
        FullValidationReport report;

        auto acyc = checkAcyclicity(schedule);
        report.acyclicity = acyc.passed;

        auto complete = checkOperationCompleteness(schedule);
        report.completeness = complete.passed;

        auto precede = checkJobPrecedence(schedule);
        report.precedence = precede.passed;

        auto machines = checkMachineSequences(schedule);
        report.machineSequences = machines.passed;

        report.overall = report.acyclicity && report.completeness &&
            report.precedence && report.machineSequences;

        if (!report.overall) {
            if (!acyc.passed) report.details = acyc.message;
            else if (!complete.passed) report.details = complete.message;
            else if (!precede.passed) report.details = precede.message;
            else if (!machines.passed) report.details = machines.message;
        }

        return report;
    }
};

class StabilityTest {
public:
    struct StabilityResult {
        double mean;
        double stdDev;
        double coefficientOfVariation;
        double minValue;
        double maxValue;
        double range;
        bool isStable;
        std::string algorithmName;

        void print() const {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "    " << algorithmName << " statistics:\n";
            std::cout << "      Mean:      " << mean << "\n";
            std::cout << "      Std Dev:   " << stdDev << "\n";
            std::cout << "      CoV:       " << coefficientOfVariation << "\n";
            std::cout << "      Min:       " << minValue << "\n";
            std::cout << "      Max:       " << maxValue << "\n";
            std::cout << "      Range:     " << range << "\n";
            std::cout << "    Stability: " << (isStable ? "STABLE (CoV < 0.05)" : "UNSTABLE (CoV >= 0.05)") << "\n";
        }
    };

    template<typename CreateAlgorithmFunc>
    StabilityResult testStability(JobShopInstance& instance,
        const std::string& name,
        int populationSize,
        int limit,
        int budget,
        int simulationRuns,
        int iterations,
        int numRuns,
        CreateAlgorithmFunc&& createAlgorithm) {

        StabilityResult result;
        result.algorithmName = name;
        std::vector<double> results;

        std::cout << "    Running " << numRuns << " " << name << " runs...\n";

        auto originalState = rng;

        for (int run = 0; run < numRuns; ++run) {
            rng.seed(12345 + run * 1000);

            auto algo = createAlgorithm();
            algo.initialize();
            algo.run(iterations);

            double bestValue = algo.getBestSolution().cachedLmaxMean;
            results.push_back(bestValue);

            if ((run + 1) % (numRuns / 2) == 0 && run > 0) {
                std::cout << "      Completed " << (run + 1) << "/" << numRuns << " runs\n";
            }
        }

        rng = originalState;

        result.mean = std::accumulate(results.begin(), results.end(), 0.0) / numRuns;
        result.minValue = *std::min_element(results.begin(), results.end());
        result.maxValue = *std::max_element(results.begin(), results.end());
        result.range = result.maxValue - result.minValue;

        double variance = 0.0;
        for (double val : results) {
            variance += std::pow(val - result.mean, 2);
        }
        variance /= (numRuns - 1);
        result.stdDev = std::sqrt(variance);
        result.coefficientOfVariation = result.stdDev / result.mean;
        result.isStable = result.coefficientOfVariation < 0.05;

        return result;
    }
};

class BoundValidator {
public:
    bool validateVarianceConsistency(const Schedule& schedule) {
        if (schedule.simulationResults.empty()) {
            std::cout << "    No simulation results available\n";
            return false;
        }

        double computedVar = schedule.getVariance();
        double n = schedule.simulationResults.size();
        double mean = schedule.cachedLmaxMean;

        double manualVar = 0.0;
        for (double val : schedule.simulationResults) {
            manualVar += (val - mean) * (val - mean);
        }
        manualVar /= (n - 1);

        bool consistent = std::abs(computedVar - manualVar) < 1e-6;
        std::cout << "    Variance consistency: " << (consistent ? "PASS" : "FAIL") << "\n";

        return consistent;
    }
};

class BaselineHeuristics {
private:
    JobShopInstance* instance;

    double evaluateSchedule(Schedule& schedule, int simRuns) {
        schedule.buildOperationSequenceFromMachineSequences(&schedule.operationSequence);
        schedule.evaluateLowerBound();
        schedule.evaluateMC(simRuns);
        return schedule.cachedLmaxMean;
    }

public:
    BaselineHeuristics(JobShopInstance* inst) : instance(inst) {}

    Schedule randomSearch(int numIterations, int simRuns) {
        Schedule best(instance);
        evaluateSchedule(best, simRuns);

        std::cout << "    Random search: ";

        for (int i = 0; i < numIterations; ++i) {
            Schedule candidate(instance);
            evaluateSchedule(candidate, simRuns);

            if (candidate.cachedLmaxMean < best.cachedLmaxMean) {
                best = candidate;
            }

            if ((i + 1) % (numIterations / 4) == 0) {
                std::cout << (i + 1) << "/" << numIterations << " ";
            }
        }

        std::cout << "Best Lmax = " << std::fixed << std::setprecision(2)
            << best.cachedLmaxMean << "\n";
        return best;
    }

    Schedule fifoSchedule(int simRuns) {
        Schedule schedule(instance);
        evaluateSchedule(schedule, simRuns);
        std::cout << "    FIFO: Lmax = " << std::fixed << std::setprecision(2)
            << schedule.cachedLmaxMean << "\n";
        return schedule;
    }

    struct HeuristicComparison {
        double randomBest;
        double fifo;
        double bestHeuristic;
        std::string bestName;

        void print() const {
            std::cout << "\n  Heuristics Results (lower is better):\n";
            std::cout << "    Random Search: " << std::fixed << std::setprecision(2) << randomBest << "\n";
            std::cout << "    FIFO:          " << fifo << "\n";
            std::cout << "    Best Heuristic: " << bestName << " = " << bestHeuristic << "\n";
        }
    };

    HeuristicComparison compareAll(int randomIterations, int simRuns) {
        HeuristicComparison result;

        Schedule bestRandom = randomSearch(randomIterations, simRuns);
        result.randomBest = bestRandom.cachedLmaxMean;

        Schedule fifo = fifoSchedule(simRuns);
        result.fifo = fifo.cachedLmaxMean;

        result.bestHeuristic = result.randomBest;
        result.bestName = "Random";

        if (result.fifo < result.bestHeuristic) {
            result.bestHeuristic = result.fifo;
            result.bestName = "FIFO";
        }

        return result;
    }
};

class GreedyABCEquivalent {
private:
    JobShopInstance* instance;
    int simulationRuns;
    int maxIterations;

    double evaluateSchedule(Schedule& schedule) {
        schedule.evaluateLowerBound();
        schedule.evaluateMC(simulationRuns);
        return schedule.cachedLmaxMean;
    }

public:
    GreedyABCEquivalent(JobShopInstance* inst, int simRuns, int maxIter)
        : instance(inst), simulationRuns(simRuns), maxIterations(maxIter) {
    }

    struct GreedyResult {
        Schedule bestSchedule;
        double bestLmax;
        std::vector<double> improvementHistory;

        void print() const {
            std::cout << "    Greedy ABC-equivalent: Best Lmax = "
                << std::fixed << std::setprecision(2) << bestLmax << "\n";
        }
    };

    GreedyResult run() {
        GreedyResult result;

        Schedule current(instance);
        evaluateSchedule(current);

        double currentBest = current.cachedLmaxMean;
        result.bestSchedule = current;
        result.bestLmax = currentBest;
        result.improvementHistory.push_back(currentBest);

        std::cout << "    Greedy search (ABC-equivalent):\n";

        bool improved = true;
        int iteration = 0;
        int noImprovementCount = 0;

        while (improved && iteration < maxIterations) {
            improved = false;

            Schedule neighbor = current.getNeighborByArcOrientation();
            evaluateSchedule(neighbor);

            if (neighbor.cachedLmaxMean < current.cachedLmaxMean) {
                current = neighbor;
                currentBest = neighbor.cachedLmaxMean;
                improved = true;
                noImprovementCount = 0;
                result.improvementHistory.push_back(currentBest);

                if (result.bestLmax > currentBest) {
                    result.bestSchedule = current;
                    result.bestLmax = currentBest;
                }

                if (result.improvementHistory.size() % 10 == 0) {
                    std::cout << "      Step " << result.improvementHistory.size()
                        << ": Lmax = " << currentBest << "\n";
                }
            } else {
                noImprovementCount++;
            }

            iteration++;

            if (noImprovementCount > 50) {
                current = Schedule(instance);
                evaluateSchedule(current);
                currentBest = current.cachedLmaxMean;
                noImprovementCount = 0;
                improved = true;
            }
        }

        std::cout << "Final Lmax: " << currentBest << "\n";
        return result;
    }
};

class GreedySearch {
private:
    JobShopInstance* instance;
    int simulationRuns;

    double evaluateSchedule(Schedule& schedule) {
        schedule.evaluateLowerBound();
        schedule.evaluateMC(simulationRuns);
        return schedule.cachedLmaxMean;
    }

public:
    enum GreedyType {
        BEST_IMPROVEMENT,
        FIRST_IMPROVEMENT,
        RANDOM_IMPROVEMENT
    };

    GreedySearch(JobShopInstance* inst, int simRuns)
        : instance(inst), simulationRuns(simRuns) {
    }

    struct GreedySearchResult {
        Schedule bestSchedule;
        double bestLmax;
        int iterations;
        int improvements;

        void print(const std::string& type) const {
            std::cout << "    " << type << ": Lmax = " << std::fixed << std::setprecision(2)
                << bestLmax << " (iterations: " << iterations
                << ")\n";
        }
    };

    GreedySearchResult bestImprovementSearch(int maxNeighbors) {
        GreedySearchResult result;
        result.iterations = 0;
        result.improvements = 0;

        Schedule current(instance);
        evaluateSchedule(current);
        result.bestSchedule = current;
        result.bestLmax = current.cachedLmaxMean;

        bool improved = true;

        while (improved && result.iterations < maxNeighbors) {
            improved = false;
            Schedule bestNeighbor = current;
            double bestValue = current.cachedLmaxMean;

            for (int i = 0; i < std::min(100, maxNeighbors - result.iterations); ++i) {
                Schedule neighbor = current.getNeighborByArcOrientation();
                evaluateSchedule(neighbor);

                if (neighbor.cachedLmaxMean < bestValue) {
                    bestNeighbor = neighbor;
                    bestValue = neighbor.cachedLmaxMean;
                    improved = true;
                }
                result.iterations++;
            }

            if (improved && bestValue < result.bestLmax) {
                current = bestNeighbor;
                result.bestSchedule = current;
                result.bestLmax = bestValue;
                result.improvements++;
            }
        }

        return result;
    }

    GreedySearchResult firstImprovementSearch(int maxIterations) {
        GreedySearchResult result;
        result.iterations = 0;
        result.improvements = 0;

        Schedule current(instance);
        evaluateSchedule(current);
        result.bestSchedule = current;
        result.bestLmax = current.cachedLmaxMean;

        for (int iter = 0; iter < maxIterations; ++iter) {
            Schedule neighbor = current.getNeighborByArcOrientation();
            evaluateSchedule(neighbor);
            result.iterations++;

            if (neighbor.cachedLmaxMean < current.cachedLmaxMean) {
                current = neighbor;
                result.improvements++;

                if (current.cachedLmaxMean < result.bestLmax) {
                    result.bestSchedule = current;
                    result.bestLmax = current.cachedLmaxMean;
                }
            }
        }

        return result;
    }
};

void addGreedyComparison(JobShopInstance& instance,
    const Schedule& sequentialBest,
    const Schedule& parallelBest,
    int simulationRuns) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GREEDY ALGORITHMS COMPARISON\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << "\n1. ABC-EQUIVALENT GREEDY (same neighborhood as ABC)\n";
    std::cout << std::string(40, '-') << "\n";

    GreedyABCEquivalent greedyABC(&instance, simulationRuns, 500);
    auto greedyResult = greedyABC.run();
    greedyResult.print();

    std::cout << "\n2. GREEDY WITH BEST IMPROVEMENT\n";
    std::cout << std::string(40, '-') << "\n";

    GreedySearch greedySearch(&instance, simulationRuns);
    auto bestImprove = greedySearch.bestImprovementSearch(500);
    bestImprove.print("Best Improvement");

    std::cout << "\n3. GREEDY WITH FIRST IMPROVEMENT\n";
    std::cout << std::string(40, '-') << "\n";

    auto firstImprove = greedySearch.firstImprovementSearch(500);
    firstImprove.print("First Improvement");

    std::cout << "\n4. COMPARISON WITH ABC ALGORITHMS\n";
    std::cout << std::string(40, '-') << "\n";

    std::cout << "  Sequential ABC:        " << std::fixed << std::setprecision(2)
        << sequentialBest.cachedLmaxMean << "\n";
    std::cout << "  Parallel ABC:          " << parallelBest.cachedLmaxMean << "\n";
    std::cout << "  Greedy ABC-equivalent: " << greedyResult.bestLmax << "\n";
    std::cout << "  Greedy Best Improve:   " << bestImprove.bestLmax << "\n";
    std::cout << "  Greedy First Improve:  " << firstImprove.bestLmax << "\n";

    double improvementOverGreedyABC = (greedyResult.bestLmax - sequentialBest.cachedLmaxMean)
        / greedyResult.bestLmax * 100;
    double improvementOverBestGreedy = (bestImprove.bestLmax - sequentialBest.cachedLmaxMean)
        / bestImprove.bestLmax * 100;

    std::cout << "\n  ABC Improvement over Greedy ABC-equiv: "
        << std::fixed << std::setprecision(2) << improvementOverGreedyABC << "%\n";
    std::cout << "  ABC Improvement over Best Greedy:      "
        << improvementOverBestGreedy << "%\n";

    std::cout << "\n5. CONCLUSION\n";
    std::cout << std::string(40, '-') << "\n";

    if (sequentialBest.cachedLmaxMean < greedyResult.bestLmax) {
        std::cout << "ABC algorithm outperforms greedy ABC-equivalent\n";
    } else if (sequentialBest.cachedLmaxMean > greedyResult.bestLmax) {
        std::cout << "Greedy ABC-equivalent outperforms ABC\n";
    } else {
        std::cout << "ABC and greedy ABC-equivalent have similar performance\n";
    }

    if (sequentialBest.cachedLmaxMean < bestImprove.bestLmax) {
        std::cout << "ABC algorithm outperforms best improvement greedy\n";
    } else if (sequentialBest.cachedLmaxMean > bestImprove.bestLmax) {
        std::cout << "Best improvement greedy outperforms ABC\n";
    } else {
        std::cout << "ABC and best improvement greedy have similar performance\n";
    }
}

void runFullValidation(JobShopInstance& instance,
    const Schedule& sequentialBest,
    const Schedule& parallelBest,
    int populationSize,
    int limit,
    int budget,
    int simulationRuns,
    int maxIterations,
    int numThreads) {

    std::cout << "\n1. STABILITY TEST (Multiple runs with different seeds)\n";
    std::cout << std::string(40, '-') << "\n";
    StabilityTest stabilityTest;

    auto seqStability = stabilityTest.testStability(
        instance, "Sequential ABC",
        populationSize, limit, budget, simulationRuns, 200, 5,
        [&]() {
            ABCAlgorithm* algo = new ABCAlgorithm(instance, populationSize, limit, budget, simulationRuns);
            return *algo;
        }
    );
    seqStability.print();

    auto parStability = stabilityTest.testStability(
        instance, "Parallel ABC (OpenMP)",
        populationSize, limit, budget, simulationRuns, 200, 5,
        [&]() {
            ParallelABCOpenMP* algo = new ParallelABCOpenMP(instance, populationSize, limit, budget, simulationRuns, numThreads);
            return *algo;
        }
    );
    parStability.print();

    std::cout << "\n2. FINAL SOLUTIONS STRUCTURE VALIDATION\n";
    std::cout << std::string(40, '-') << "\n";

    AlgorithmValidator validator;
    BoundValidator boundValidator;

    std::cout << "\n  Sequential ABC:\n";
    auto seqReport = validator.validateSchedule(sequentialBest);
    seqReport.print();

    boundValidator.validateVarianceConsistency(sequentialBest);

    std::cout << "\n  Parallel ABC (OpenMP):\n";
    auto parReport = validator.validateSchedule(parallelBest);
    parReport.print();

    boundValidator.validateVarianceConsistency(parallelBest);

    std::cout << "\n3. BASELINE HEURISTICS (FIFO & Random Search)\n";
    std::cout << std::string(40, '-') << "\n";
    BaselineHeuristics heuristics(&instance);
    auto heuristicResults = heuristics.compareAll(100, simulationRuns);
    heuristicResults.print();

    std::cout << "\n4. COMPARISON WITH BASELINES\n";
    std::cout << std::string(40, '-') << "\n";

    auto checkImprovement = [](double abcResult, double heuristicBest, const std::string& name) {
        bool improved = abcResult < heuristicBest;
        double diff = heuristicBest - abcResult;
        double diffPercent = diff / heuristicBest * 100.0;

        std::cout << "  " << name << " ABC vs Best Heuristic: "
            << std::fixed << std::setprecision(2) << abcResult
            << " vs " << heuristicBest;

        if (improved) {
            std::cout << " : BETTER by " << std::abs(diffPercent) << "%\n";
        } else {
            std::cout << " : WORSE by " << std::abs(diffPercent) << "%\n";
        }

        return improved;
        };

    bool seqImproved = checkImprovement(sequentialBest.cachedLmaxMean,
        heuristicResults.bestHeuristic,
        "Sequential");
    bool parImproved = checkImprovement(parallelBest.cachedLmaxMean,
        heuristicResults.bestHeuristic,
        "Parallel");

    std::cout << "\n5. SEQUENTIAL VS PARALLEL COMPARISON\n";
    std::cout << std::string(40, '-') << "\n";

    double diff = (parallelBest.cachedLmaxMean - sequentialBest.cachedLmaxMean)
        / sequentialBest.cachedLmaxMean;

    std::cout << "  Sequential Lmax: " << std::fixed << std::setprecision(2)
        << sequentialBest.cachedLmaxMean << "\n";
    std::cout << "  Parallel Lmax:   " << parallelBest.cachedLmaxMean << "\n";
    std::cout << "  Difference:      " << diff << " ";

    if (parallelBest.cachedLmaxMean < sequentialBest.cachedLmaxMean) {
        std::cout << "(Parallel is better)\n";
    } else if (sequentialBest.cachedLmaxMean < parallelBest.cachedLmaxMean) {
        std::cout << "(Sequential is better)\n";
    } else {
        std::cout << "(Tie)\n";
    }

    std::cout << "\n6. LOWER BOUND QUALITY ASSESSMENT\n";
    std::cout << std::string(40, '-') << "\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "VALIDATION SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << "================================= STRUCTURAL TESTS ===============================\n";
    std::cout << "\n";
    std::cout << "Schedule Acyclicity: " << (seqReport.acyclicity ? "PASS" : "FAIL") << "\n";
    std::cout << "Schedule Completeness         : " << (seqReport.completeness ? "PASS" : "FAIL") << "\n";
    std::cout << "Job Precedence                : " << (seqReport.precedence ? "PASS" : "FAIL") << "\n";
    std::cout << "Machine Sequences             : " << (seqReport.machineSequences ? "PASS" : "FAIL") << "\n";
    std::cout << "\n";
    std::cout << "================================= PERFORMANCE TESTS ===============================\n";
    std::cout << "\n";
    std::cout << "Sequential vs Heuristics      : " << (seqImproved ? "PASS" : "FAIL") << " \n";
    std::cout << "Parallel vs Heuristics        : " << (parImproved ? "PASS" : "FAIL") << "\n";
    std::cout << "Sequential vs Parallel        : " << (parallelBest.cachedLmaxMean < sequentialBest.cachedLmaxMean ? "Parallel wins" : "Sequential wins") << "\n";
    std::cout << "\n";

    addGreedyComparison(instance, sequentialBest, parallelBest, simulationRuns);
}

#endif