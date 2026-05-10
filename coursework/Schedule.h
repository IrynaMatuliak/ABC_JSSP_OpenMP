#ifndef SCHEDULE_H
#define SCHEDULE_H

#include "JobShopInstance.h"
#include <vector>
#include <set>
#include <array>
#include <unordered_map>

class Schedule {
public:
    std::vector<std::vector<int>> machineSequences;
    std::vector<int> operationSequence;
    JobShopInstance* instance;
    double cachedLmaxMean;
    int cachedLmaxLBJob;
    double cachedLmaxLB;
    double cachedVariance;
    int evalCount;
    std::vector<double> simulationResults;

    Schedule();
    Schedule(JobShopInstance* inst);
    Schedule(const Schedule& other);
    Schedule& operator=(const Schedule& other);
    ~Schedule() = default;

    bool isAcyclicFast() const;
    bool verifyFullAcyclicity() const;

    void repair();
    void initializeAcyclic();

    std::pair<int, double> evaluateDeterministic(const std::vector<double>& procTimes) const;
    std::pair<int, double> evaluateLowerBound() const;
    double evaluateMC(int numRepl);
    double evaluateMCParallel(int numRepl, int numThreads);
    double getVariance() const { return cachedVariance; }

    Schedule getNeighbor() const;
    Schedule getNeighborFast_MoveBackward() const;
    Schedule getNeighborFast_Swap() const;
    Schedule getNeighborFast() const;
    Schedule getNeighborByBlock() const;
    Schedule getNeighborByArcOrientation() const;

    bool buildOperationSequenceFromMachineSequences(std::vector<int>* opSequence) const;

    bool operator<(const Schedule& other) const;

    void printSummary() const;
    void printDetailedSchedule() const;

    bool validateConsistency() const;

    void printOperationSequence(const std::string& filename) const;
    void printMachineGanttChart(const std::string& filename) const;

private:
    double singleSimulation() const;
    void updateStatistics();
    int getOperationId(int job, int step) const { return job * instance->numMachines + step; }
};

#endif