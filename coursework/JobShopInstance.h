#ifndef JOBSHOPINSTANCE_H
#define JOBSHOPINSTANCE_H

#include <vector>
#include <string>
#include <random>

struct Operation {
    int job;
    int machine;
    double mean;
    double variance;
    int jobSeqIdx;

    Operation(int j, int m, double mn, double var, int idx)
        : job(j), machine(m), mean(mn), variance(var), jobSeqIdx(idx) {
    }

    int getOperationId() const { return job * 1000 + jobSeqIdx; }
};

class JobShopInstance {
public:
    enum DistType { NORMAL, UNIFORM, EXPONENTIAL, MEAN };

    int numJobs;
    int numMachines;
    std::vector<std::vector<Operation>> jobs;
    std::vector<double> dueDates;
    std::vector<int> jobWeights;
    DistType distType;
    double theta;

    std::vector<std::vector<int>> operationsPerMachine;

    JobShopInstance(int nJobs, int nMachines, DistType dist, double thetaVal);
    void generateRandomInstance(unsigned seed);
    void generateRealisticDueDates(int numSimulations = 100);
    void generateDueDates(int numSimulations = 100);
    double sampleProcessingTime(const Operation& op) const;
    double getMeanProcessingTime(const Operation& op) const;

    int getOperationIndex(int job, int step) const { return job * numMachines + step; }
    void getJobAndStep(int opIdx, int& job, int& step) const {
        job = opIdx / numMachines;
        step = opIdx % numMachines;
    }


    void writeToFile(const std::string& fileName);
    void readFromFile(const std::string& fileName);

private:
    JobShopInstance();

    void generateRandomRoutes();
    void generateRandomProcessingMeans();
    void buildOperationsPerMachine();
};

#endif