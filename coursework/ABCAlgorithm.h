#ifndef ABCALGORITHM_H
#define ABCALGORITHM_H

#include "JobShopInstance.h"
#include "Schedule.h"
#include <vector>

class ABCAlgorithm {
protected:
    const JobShopInstance& instance;
    int SN;
    int limit;
    int T;
    int maxIterations;
    int simulationRuns;
    std::vector<Schedule> population;
    std::vector<int> trials;
    std::vector<double> fitness;

    Schedule bestSolution;

    const int requiredSimulations = 10;

    double calculateFitness(const Schedule& s) const;
    virtual void updateFitnessValues();
    int rouletteWheelSelection() const;

    bool isSignificantlyBetter(const Schedule& a, const Schedule& b, double alpha = 0.05);
    void allocateBudget(std::vector<Schedule>& solutions, int totalBudget);
    void preScreenAndEvaluate(Schedule& newSol, const Schedule& currentSol);

    const Schedule& getBestSolutionInPopulation() const;

public:
    ABCAlgorithm(const JobShopInstance& inst, int sn, int lim, int t, int simulationRuns);
    virtual ~ABCAlgorithm() = default;

    virtual void initialize();
    virtual void run(int maxIter);

    const Schedule& getBestSolution() const;
    
    double getBestFitness() const;
};

#endif