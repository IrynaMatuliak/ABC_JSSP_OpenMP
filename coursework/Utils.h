#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

extern std::mt19937 rng;

inline double randDouble(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

inline int randInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

inline double normalSample(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(rng);
}

inline double uniformSample(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

inline double exponentialSample(double lambda) {
    std::exponential_distribution<double> dist(lambda);
    return dist(rng);
}

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsedSec() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_time).count();
    }
};

#endif