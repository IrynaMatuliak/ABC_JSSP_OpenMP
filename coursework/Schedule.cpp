#include "Schedule.h"
#include "Utils.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <array>
#include <queue>
#include <set>
#include <omp.h>
#include <random>
#include <chrono>
#include <fstream>
#include <cassert>

static thread_local std::mt19937 local_rng(std::random_device{}());
static thread_local bool thread_seeded = false;

Schedule::Schedule()
    : instance(nullptr)
    , cachedLmaxMean(1e9)
    , cachedLmaxLBJob(-1)
    , cachedLmaxLB(1e9)
    , cachedVariance(0.0)
    , evalCount(0) {
}

Schedule::Schedule(const Schedule& other) {
    instance = other.instance;
    machineSequences = other.machineSequences;
    operationSequence = other.operationSequence;
    cachedLmaxMean = other.cachedLmaxMean;
    cachedLmaxLBJob = other.cachedLmaxLBJob;
    cachedLmaxLB = other.cachedLmaxLB;
    cachedVariance = other.cachedVariance;
    evalCount = other.evalCount;
    simulationResults = other.simulationResults;
}

Schedule& Schedule::operator=(const Schedule& other) {
    if (this != &other) {
        instance = other.instance;
        machineSequences = other.machineSequences;
        operationSequence = other.operationSequence;
        cachedLmaxMean = other.cachedLmaxMean;
        cachedLmaxLBJob = other.cachedLmaxLBJob;
        cachedLmaxLB = other.cachedLmaxLB;
        cachedVariance = other.cachedVariance;
        evalCount = other.evalCount;
        simulationResults = other.simulationResults;
    }
    return *this;
}

Schedule::Schedule(JobShopInstance* inst)
    : instance(inst)
    , cachedLmaxMean(1e9)
    , cachedLmaxLBJob(-1)
    , cachedLmaxLB(1e9)
    , cachedVariance(0.0)
    , evalCount(0) {

    if (instance == nullptr) return;

    int numJobs = instance->numJobs;
    int numMachines = instance->numMachines;

    machineSequences.resize(numMachines);
    for (int k = 0; k < numMachines; ++k) {
        machineSequences[k] = instance->operationsPerMachine[k];
    }

    buildOperationSequenceFromMachineSequences(&operationSequence);
}

bool Schedule::isAcyclicFast() const {
    return buildOperationSequenceFromMachineSequences(nullptr);
}

bool Schedule::verifyFullAcyclicity() const {
    if (instance == nullptr) return false;

    int n = instance->numJobs;
    int m = instance->numMachines;
    int totalOps = n * m;

    std::vector<std::vector<int>> adj(totalOps);
    std::vector<int> inDegree(totalOps, 0);

    for (int j = 0; j < n; ++j) {
        for (int s = 0; s < m - 1; ++s) {
            int from = j * m + s;
            int to = j * m + s + 1;
            adj[from].push_back(to);
            inDegree[to]++;
        }
    }

    for (int k = 0; k < m; ++k) {
        const auto& seq = machineSequences[k];
        for (size_t i = 0; i < seq.size(); ++i) {
            for (size_t j = i + 1; j < seq.size(); ++j) {
                int from = seq[i];
                int to = seq[j];
                adj[from].push_back(to);
                inDegree[to]++;
            }
        }
    }

    std::queue<int> q;
    for (int i = 0; i < totalOps; ++i) {
        if (inDegree[i] == 0) q.push(i);
    }

    int count = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        count++;
        for (int v : adj[u]) {
            if (--inDegree[v] == 0) q.push(v);
        }
    }

    return count == totalOps;
}

bool Schedule::buildOperationSequenceFromMachineSequences(std::vector<int>* opSequence) const {
   if (instance == nullptr) return false;

   int numJobs = instance->numJobs;
   int numMachines = instance->numMachines;

   if (opSequence) {
      opSequence->clear();
      opSequence->reserve(numJobs * numMachines);
   }

   std::vector<int> stepsOfJob(numJobs, 0);
   std::vector<int> stepsOnMachine(numMachines, 0);

   int idleMachines = 0;
   while (idleMachines < numMachines) {
      idleMachines = 0;
      int addedOperations = 0;
      for (int machineIdx = 0; machineIdx < numMachines; ++machineIdx) {
         const auto& operationsOnMachine = machineSequences[machineIdx];

         if (stepsOnMachine[machineIdx] >= operationsOnMachine.size()) {
            idleMachines++;
         }
         else {
            const int operationId = operationsOnMachine[stepsOnMachine[machineIdx]];
            const int jobId = operationId / numMachines;
            const int jobStepId = operationId % numMachines;

            if (stepsOfJob[jobId] == jobStepId) {
               addedOperations++;
               stepsOfJob[jobId]++;
               stepsOnMachine[machineIdx]++;
               if (opSequence) 
                  opSequence->push_back(operationId);
            }
         }
      }
      if (idleMachines < numMachines && addedOperations == 0) {
         return false;
      }
   }
   
   return true;
}

void Schedule::repair() {
    if (instance == nullptr) return;

    int m = instance->numMachines;

    for (int k = 0; k < m; ++k) {
        std::sort(machineSequences[k].begin(), machineSequences[k].end());
    }

    buildOperationSequenceFromMachineSequences(&operationSequence);
}

void Schedule::initializeAcyclic() {
    repair();
}

namespace {
   bool checkConsistency_MoveBefore(const Schedule& schedule, int operationBefore, int operationToMove) {
      if (!schedule.instance)
         return false;

      const int numMachines = schedule.instance->numMachines;
      const int jobForOp1 = operationToMove / numMachines;
      const int jobForOp2 = operationBefore / numMachines;

      // same job - cannot swap
      if (jobForOp1 == jobForOp2) return false;

      bool isBrokenSequence = false;
      auto itStart = std::find(schedule.operationSequence.begin(), schedule.operationSequence.end(), operationBefore);
      for (auto it = itStart + 1; it != schedule.operationSequence.end(); ++it) {
         if (*it == operationToMove)
            break; // end of the range. no inconsistencies found.
         const int itJobId = *it / numMachines;
         if (itJobId == jobForOp1) {
            isBrokenSequence = true; // found the operation, inside the given range, belongs to the job to move. Operation cannot be permitted.
            break;
         }
      }

      return !isBrokenSequence;
   }

   bool checkConsistency_Swap(const Schedule& schedule, int operation1, int operation2) {
      if (!schedule.instance) 
         return false;

      const int numMachines = schedule.instance->numMachines;
      const int jobForOp1   = operation1 / numMachines;
      const int jobForOp2   = operation2 / numMachines;

      // same job - cannot swap
      if (jobForOp1 == jobForOp2) return false;

      bool isBrokenSequence = false;
      auto itStart = std::find(schedule.operationSequence.begin(), schedule.operationSequence.end(), operation1);
      for (auto it = itStart + 1; it != schedule.operationSequence.end(); ++it) {
         if (*it == operation2)
            break; // end of the range. no inconsistencies found.
         const int itJobId = *it / numMachines;
         if (itJobId == jobForOp1 || itJobId == jobForOp2) {
            isBrokenSequence = true; // found the operation, inside the given range, belongs to one of the job to swap. Operation cannot be permitted.
            break;
         }
      }

      return !isBrokenSequence;
   }
}

// move operation of a job with max lateness backward
Schedule Schedule::getNeighborFast_MoveBackward() const {
   const int MAX_ATTEMPTS = 30;

   struct STestedSwap {
      int machine;
      int op1;
      int op2;
   };
   std::vector<STestedSwap> testedPairs;

   for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      Schedule nb(*this);
      const int numMachines = instance->numMachines;

      int machine = randInt(0, numMachines - 1);

      if (nb.machineSequences[machine].size() >= 2) {
         const size_t len = nb.machineSequences[machine].size();

         int opIdx1 = 0;
         int opIdx2 = 1;
         if (cachedLmaxLBJob == -1) {
            opIdx1 = randInt(0, len - 2);
            opIdx2 = opIdx1 + 1;
         }
         else {
            const auto& operations = nb.machineSequences[machine];
            auto it = std::find_if(operations.rbegin(), operations.rend(), [&](auto& item) { return item / numMachines == cachedLmaxLBJob;});
            if (it != operations.rend()) {
               // need to move job with max lateness to front
               opIdx2 = std::distance(&*operations.begin(), &*it);
               opIdx1 = opIdx2 == 0 ? 0 : randInt(0, opIdx2 - 1);
            }
         }

         if (opIdx1 == opIdx2)
            continue;

         testedPairs.push_back({ machine, opIdx1, opIdx2 });

         const int op1 = nb.machineSequences[machine][opIdx1];
         const int op2 = nb.machineSequences[machine][opIdx2];
         if (checkConsistency_MoveBefore(nb, op1, op2)) {
            for (int i = opIdx2; i > opIdx1; --i) {
               std::swap(nb.machineSequences[machine][i-1], nb.machineSequences[machine][i]);
            }
            if (nb.buildOperationSequenceFromMachineSequences(&nb.operationSequence)) {
               nb.cachedLmaxMean = 1e9;
               nb.cachedLmaxLBJob = -1;
               nb.cachedLmaxLB = 1e9;
               nb.cachedVariance = 0.0;
               nb.evalCount = 0;
               nb.simulationResults.clear();
               return nb;
            }
            else {
               // swap it back because topology builder has failed in despite the fact that consistency is OK...
               std::swap(nb.machineSequences[machine][opIdx1], nb.machineSequences[machine][opIdx2]);
            }
         }
      }
   }

   Schedule nb(*this);
   nb.buildOperationSequenceFromMachineSequences(&nb.operationSequence);
   return nb;
}

Schedule Schedule::getNeighborFast_Swap() const {
   const int MAX_ATTEMPTS = 30;

   struct STestedSwap {
      int machine;
      int op1;
      int op2;
   };
   std::vector<STestedSwap> testedPairs;

   for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      Schedule nb(*this);
      const int numMachines = instance->numMachines;

      int machine = randInt(0, numMachines - 1);

      if (nb.machineSequences[machine].size() >= 2) {
         const size_t len = nb.machineSequences[machine].size();

         int opIdx1 = 0;
         int opIdx2 = 1;
         if (cachedLmaxLBJob == -1) {
            opIdx1 = randInt(0, len - 2);
            opIdx2 = opIdx1 + 1;
         }
         else {
            const auto& operations = nb.machineSequences[machine];
            auto it = std::find_if(operations.begin(), operations.end(), [&](auto& item) { return item / numMachines == cachedLmaxLBJob;});
            if (it != operations.end()) {
               // need to move job with max lateness to front
               opIdx2 = std::distance(operations.begin(), it);
               opIdx1 = opIdx2 == 0 ? 0 : randInt(0, opIdx2 - 1);
            }
         }

         if (opIdx1 == opIdx2)
            continue;

         testedPairs.push_back({ machine, opIdx1, opIdx2 });

         const int op1 = nb.machineSequences[machine][opIdx1];
         const int op2 = nb.machineSequences[machine][opIdx2];
         if (checkConsistency_Swap(nb, op1, op2)) {
            std::swap(nb.machineSequences[machine][opIdx1], nb.machineSequences[machine][opIdx2]);
            if (nb.buildOperationSequenceFromMachineSequences(&nb.operationSequence)) {
               nb.cachedLmaxMean = 1e9;
               nb.cachedLmaxLBJob = -1;
               nb.cachedLmaxLB = 1e9;
               nb.cachedVariance = 0.0;
               nb.evalCount = 0;
               nb.simulationResults.clear();
               return nb;
            }
            else {
               // swap it back because topology builder has failed in despite the fact that consistency is OK...
               std::swap(nb.machineSequences[machine][opIdx1], nb.machineSequences[machine][opIdx2]);
            }
         }
      }
   }

   Schedule nb(*this);
   nb.buildOperationSequenceFromMachineSequences(&nb.operationSequence);
   return nb;
}

Schedule Schedule::getNeighborFast() const
{
   //return getNeighborFast_Swap();
   return getNeighborFast_MoveBackward();
}

Schedule Schedule::getNeighbor() const {
    return getNeighborFast();
}

Schedule Schedule::getNeighborByBlock() const {
    return getNeighborFast();
}

Schedule Schedule::getNeighborByArcOrientation() const {
    return getNeighborFast();
}

namespace {
   double calculateProcessingTime(JobShopInstance::DistType distType, double mean, double theta) {

      double processingTime = 0.0;
      switch (distType) {
      case JobShopInstance::MEAN: {
         processingTime = mean;
         break;
      }
      case JobShopInstance::NORMAL: {
         std::normal_distribution<double> dist(mean, theta * mean);
         processingTime = std::max(0.1, dist(local_rng));
         break;
      }
      case JobShopInstance::UNIFORM: {
         double range = theta * mean;
         std::uniform_real_distribution<double> dist(mean - range, mean + range);
         processingTime = std::max(0.1, dist(local_rng));
         break;
      }
      case JobShopInstance::EXPONENTIAL: {
         std::exponential_distribution<double> dist(1.0 / mean);
         processingTime = dist(local_rng);
         break;
      }
      }

      return processingTime;
   }
}

double Schedule::singleSimulation() const {
    if (instance == nullptr) return 1e9;

    int n = instance->numJobs;
    int m = instance->numMachines;
    int totalOps = n * m;

    std::vector<double> procTimes(totalOps);
    std::vector<double> jobNextAvail(n, 0.0);
    std::vector<double> machNextAvail(m, 0.0);

    for (int j = 0; j < n; ++j) {
        for (int s = 0; s < m; ++s) {
            const int idx = getOperationId(j, s);
            const Operation& op = instance->jobs[j][s];
            procTimes[idx] = calculateProcessingTime(instance->distType, op.mean, instance->theta);
        }
    }

    for (int opIdx : operationSequence) {
        int job = opIdx / m;
        int step = opIdx % m;
        int machine = instance->jobs[job][step].machine;
        double duration = procTimes[opIdx];
        double start = std::max(jobNextAvail[job], machNextAvail[machine]);
        double end = start + duration;

        jobNextAvail[job] = end;
        machNextAvail[machine] = end;
    }

    double maxLateness = -1e9;
    for (int j = 0; j < n; ++j) {
        const double lateness = jobNextAvail[j] - instance->dueDates[j];
        maxLateness = std::max(lateness, maxLateness);
    }
    return maxLateness;
}

double Schedule::evaluateMC(int numRepl) {
    simulationResults.clear();
    simulationResults.reserve(numRepl);

    for (int r = 0; r < numRepl; ++r) {
        simulationResults.push_back(singleSimulation());
    }

    updateStatistics();
    return cachedLmaxMean;
}

double Schedule::evaluateMCParallel(int numRepl, int numThreads) {
    if (numRepl <= 0) return cachedLmaxMean;

    std::vector<double> results(numRepl);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
    for (int r = 0; r < numRepl; ++r) {
        results[r] = singleSimulation();
    }

    simulationResults = std::move(results);
    updateStatistics();
    return cachedLmaxMean;
}

void Schedule::updateStatistics() {
    if (simulationResults.empty()) return;

    const double sum = std::accumulate(simulationResults.begin(), simulationResults.end(), 0.0);
    cachedLmaxMean = sum / simulationResults.size();

    double sumSq = 0.0;
    for (double val : simulationResults) {
        double diff = val - cachedLmaxMean;
        sumSq += diff * diff;
    }
    cachedVariance = sumSq / (simulationResults.size() - 1);
    evalCount = simulationResults.size();
}

/**
* returns the pair where
* first - ID of a job with Lmax
* second - average Lmax of all jobs with positive Lmax.
*/
std::pair<int, double> Schedule::evaluateDeterministic(const std::vector<double>& procTimes) const {
   if (instance == nullptr) return { -1, 1e9 };

   int n = instance->numJobs;
   int m = instance->numMachines;
   std::vector<double> jobNextAvail(n, 0.0);
   std::vector<double> machNextAvail(m, 0.0);
   std::vector<double> jobCompletion(n, 0.0);

   for (int opIdx : operationSequence) {
      int job = opIdx / m;
      int step = opIdx % m;
      int machine = instance->jobs[job][step].machine;
      double duration = procTimes[opIdx];
      double start = std::max(jobNextAvail[job], machNextAvail[machine]);
      double end = start + duration;

      jobNextAvail[job] = end;
      machNextAvail[machine] = end;

      if (step == m - 1) {
         jobCompletion[job] = end;
      }
   }

   double sumLateness = 0.0;
   double maxLateness = -1e9;
   int maxLatenessJob = -1;
   for (int j = 0; j < n; ++j) {
      double lateness = jobCompletion[j] - instance->dueDates[j];
      sumLateness += std::max(0.0, lateness);
      if (lateness > maxLateness) {
         maxLateness = lateness;
         maxLatenessJob = j;
      }
   }
   return { maxLatenessJob, /*sumLateness / n*/ maxLateness };
}

/**
* returns the pair where
* first - ID of a job with Lmax
* second - average Lmax of all jobs with positive Lmax.
*/
std::pair<int, double> Schedule::evaluateLowerBound() const {
    if (instance == nullptr) return { -1, 1e9 };

    int totalOps = instance->numJobs * instance->numMachines;
    std::vector<double> meanTimes(totalOps);
    for (int j = 0; j < instance->numJobs; ++j) {
        for (int s = 0; s < instance->numMachines; ++s) {
            int idx = getOperationId(j, s);
            meanTimes[idx] = instance->jobs[j][s].mean;
        }
    }
    return evaluateDeterministic(meanTimes);
}

bool Schedule::operator<(const Schedule& other) const {
    return cachedLmaxMean < other.cachedLmaxMean;
}

void Schedule::printSummary() const {
    std::cout << "Lmax=" << std::fixed << std::setprecision(2) << cachedLmaxMean
        << ", LB=" << cachedLmaxLB
        << ", Eval=" << evalCount
        << ", Feasible=" << (isAcyclicFast() ? "Yes" : "No") << std::endl;
}

void Schedule::printDetailedSchedule() const {
    printSummary();
}

bool Schedule::validateConsistency() const {
    return isAcyclicFast();
}

void Schedule::printOperationSequence(const std::string& filename) const {
    if (instance == nullptr) {
        std::cout << "No instance available.\n";
        return;
    }

    std::ofstream outFile(filename, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    int totalOps = instance->numJobs * instance->numMachines;
    outFile << "\nOperation sequence (total: " << totalOps << " operations):\n";

    int count = 0;
    for (int opIdx : operationSequence) {
        int job = opIdx / instance->numMachines;
        int step = opIdx % instance->numMachines;
        outFile << "J" << job << "O" << step;
        if (++count % 10 == 0 && count < (int)operationSequence.size())
            outFile << " ->\n";
        else if (count < (int)operationSequence.size())
            outFile << " -> ";
    }
    outFile << "\n";
    outFile.close();
}

void Schedule::printMachineGanttChart(const std::string& filename) const {
    if (instance == nullptr) {
        std::cout << "No instance available.\n";
        return;
    }

    std::ofstream outFile(filename, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    int n = instance->numJobs;
    int m = instance->numMachines;
    int totalOps = n * m;

    std::vector<double> meanTimes(totalOps);
    for (int j = 0; j < n; ++j) {
        for (int s = 0; s < m; ++s) {
            meanTimes[j * m + s] = instance->jobs[j][s].mean;
        }
    }

    std::vector<double> jobNextAvail(n, 0.0);
    std::vector<double> machNextAvail(m, 0.0);
    std::vector<int> jobStep(n, 0);

    struct OpInfo {
        int job;
        int step;
        int machine;
        double start;
        double end;
    };
    std::vector<OpInfo> opInfos(totalOps);

    for (int opIdx : operationSequence) {
        int job = opIdx / m;
        int step = jobStep[job];
        int machine = instance->jobs[job][step].machine;
        double duration = meanTimes[opIdx];
        double start = std::max(jobNextAvail[job], machNextAvail[machine]);
        double end = start + duration;

        opInfos[opIdx] = { job, step, machine, start, end };

        jobNextAvail[job] = end;
        machNextAvail[machine] = end;
        jobStep[job]++;
    }

    std::vector<std::vector<OpInfo>> machineOps(m);
    for (int opIdx = 0; opIdx < totalOps; ++opIdx) {
        const auto& info = opInfos[opIdx];
        machineOps[info.machine].push_back(info);
    }

    for (int k = 0; k < m; ++k) {
        std::sort(machineOps[k].begin(), machineOps[k].end(),
            [](const OpInfo& a, const OpInfo& b) { return a.start < b.start; });
    }

    outFile << "\nGantt Chart (deterministic using mean processing times):\n";
    for (int k = 0; k < m; ++k) {
        outFile << "Machine " << std::setw(2) << k << ":";
        for (const auto& op : machineOps[k]) {
            outFile << " |J" << std::setw(2) << op.job
                << " [" << std::fixed << std::setprecision(1) << std::setw(6) << op.start
                << "-" << std::setw(6) << op.end << "]";
        }
        outFile << " |\n";
    }
    outFile << "\n";


    std::vector<double> machineLoad(m, 0.0);
    std::vector<double> machineIdle(m, 0.0);
    for (int machineIdx = 0; machineIdx < machineOps.size(); ++machineIdx) {
       const auto& operations = machineOps[machineIdx];
       for (int opIdx = 1; opIdx < operations.size(); ++opIdx) {
          machineIdle[machineIdx] += operations[opIdx].start - operations[opIdx - 1].end;
       }
       for (int opIdx = 0; opIdx < operations.size(); ++opIdx) {
          machineLoad[machineIdx] += operations[opIdx].end - operations[opIdx].start;
       }
    }

    outFile << "\nMachines load efficiency:\n";
    for (int machineIdx = 0; machineIdx < m; ++machineIdx) {
       outFile << "Machine " << std::setw(2) << machineIdx << ":";
       outFile << std::fixed << std::setw(6) << std::setprecision(2) 
          << " | Load = " << machineLoad[machineIdx]
          << " | Idle = " << machineIdle[machineIdx] 
          << std::endl;
    }
    outFile << "\n";

    outFile.close();
}