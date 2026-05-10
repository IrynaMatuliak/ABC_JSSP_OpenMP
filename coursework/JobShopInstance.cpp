#include "JobShopInstance.h"
#include "Utils.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"

JobShopInstance::JobShopInstance()
   : numJobs(0), numMachines(0), distType(DistType::NORMAL), theta(0.0) {
   jobs.resize(0);
   dueDates.resize(0);
   jobWeights.resize(0);
}

JobShopInstance::JobShopInstance(int nJobs, int nMachines, DistType dist, double thetaVal)
    : numJobs(nJobs), numMachines(nMachines), distType(dist), theta(thetaVal) {
    jobs.resize(numJobs);
    dueDates.resize(numJobs, 0.0);
    jobWeights.resize(numJobs, 1);
}

void JobShopInstance::generateRandomRoutes() {
    for (int j = 0; j < numJobs; ++j) {
        std::vector<int> machOrder(numMachines);
        for (int i = 0; i < numMachines; ++i) machOrder[i] = i;
        std::shuffle(machOrder.begin(), machOrder.end(), rng);
        jobs[j].clear();
        for (int k = 0; k < numMachines; ++k) {
            jobs[j].emplace_back(j, machOrder[k], 0.0, 0.0, k);
        }
    }
    buildOperationsPerMachine();
}

void JobShopInstance::buildOperationsPerMachine() {
   operationsPerMachine.resize(numMachines);
   for (int k = 0; k < numMachines; ++k) {
      operationsPerMachine[k].reserve(numJobs);
   }

   for (int operationIdx = 0; operationIdx < numMachines; ++operationIdx) {
      for (int jobIdx = 0; jobIdx < numJobs; ++jobIdx) {
         const int machineId = jobs[jobIdx][operationIdx].machine;
         operationsPerMachine[machineId].push_back(jobIdx * numMachines + operationIdx);
      }
   }
}

void JobShopInstance::generateRandomProcessingMeans() {
    for (int j = 0; j < numJobs; ++j) {
        for (int k = 0; k < numMachines; ++k) {
            double mean = randDouble(1.0, 99.0);
            jobs[j][k].mean = mean;
            jobs[j][k].variance = theta * mean;
        }
    }
}

void JobShopInstance::generateDueDates(int numSimulations) {

   std::vector<double> totalJobTime(numJobs, 0.0);
   std::vector<double> machineOneJobTime(numJobs, 0.0);

   for (size_t jobIdx = 0; jobIdx < jobs.size(); ++jobIdx) {
      auto& job = jobs[jobIdx];
      for (auto& op : job) {
         totalJobTime[jobIdx] += op.mean;
         if (op.machine == 1) {
            machineOneJobTime[jobIdx] = (jobIdx == 0 ? 0 : machineOneJobTime[jobIdx-1]) + op.mean;
         }
      }
      totalJobTime[jobIdx] *= randDouble(1.0, 1.5);
   }

   dueDates[0] = totalJobTime[0];

   for (size_t jobIdx = 1; jobIdx < numJobs; ++jobIdx) {
      dueDates[jobIdx] = dueDates[jobIdx - 1] + totalJobTime[jobIdx];
   }
   std::shuffle(dueDates.begin(), dueDates.end(), rng);
}

void JobShopInstance::generateRealisticDueDates(int numSimulations) {
    std::vector<double> completionTimes(numJobs, 0.0);

    for (int sim = 0; sim < numSimulations; ++sim) {
        std::vector<double> jobTimes(numJobs, 0.0);
        std::vector<double> machTimes(numMachines, 0.0);

        for (int op = 0; op < numJobs * numMachines; ++op) {
            int bestJob = -1;
            double bestPriority = -1e9;

            for (int j = 0; j < numJobs; ++j) {
                int step = 0;
                while (step < numMachines && jobTimes[j] > machTimes[jobs[j][step].machine]) {
                    step++;
                }
                if (step < numMachines) {
                    double priority = -jobTimes[j];
                    if (priority > bestPriority) {
                        bestPriority = priority;
                        bestJob = j;
                    }
                }
            }

            if (bestJob >= 0) {
                int step = 0;
                while (step < numMachines && jobTimes[bestJob] > machTimes[jobs[bestJob][step].machine]) {
                    step++;
                }
                int machine = jobs[bestJob][step].machine;
                double duration = sampleProcessingTime(jobs[bestJob][step]);
                double start = std::max(jobTimes[bestJob], machTimes[machine]);
                double end = start + duration;
                jobTimes[bestJob] = end;
                machTimes[machine] = end;

                if (step == numMachines - 1) {
                    completionTimes[bestJob] += end;
                }
            }
        }
    }

    for (int j = 0; j < numJobs; ++j) {
        dueDates[j] = completionTimes[j] / numSimulations * randDouble(0.8, 1.2);
    }
}

void writeOperationToFile(const Operation& op, rapidjson::Writer<rapidjson::StringBuffer>& writer)
{
   writer.StartObject();

   writer.Key("job");
   writer.Int(op.job);
   writer.Key("machine");
   writer.Int(op.machine);
   writer.Key("mean");
   writer.Double(op.mean);
   writer.Key("variance");
   writer.Double(op.variance);
   writer.Key("jobSeqIdx");
   writer.Int(op.jobSeqIdx);

   writer.EndObject();
}

void JobShopInstance::writeToFile(const std::string& fileName)
{
   std::ofstream outfile;

   outfile.open(fileName, std::ios::out);
   if (outfile.is_open()) {
      rapidjson::StringBuffer s;
      rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(s);

      writer.SetMaxDecimalPlaces(4);

      writer.StartObject();
      writer.Key("jobs-count");
      writer.Int(numJobs);
      writer.Key("machines-count");
      writer.Int(numMachines);

      writer.Key("distance-type");
      writer.Int(int(distType));

      writer.Key("theta");
      writer.Double(theta);

      writer.Key("jobs");
      writer.StartArray();
      for (size_t jobId = 0u; jobId < jobs.size(); ++jobId) {
         writer.StartObject();
         writer.Key("jobId");
         writer.Int(int(jobId));
         writer.Key("operations");
         writer.StartArray();
         for (auto& op : jobs[jobId]) {
            ::writeOperationToFile(op, writer);
         }
         writer.EndArray();
         writer.EndObject();
      }
      writer.EndArray();

      writer.Key("duedates");
      writer.StartArray();
      for (double& item: dueDates) {
         writer.Double(item);
      }
      writer.EndArray();

      writer.Key("job-weights");
      writer.StartArray();
      for (int& item : jobWeights) {
         writer.Int(item);
      }
      writer.EndArray();

      writer.EndObject();

      outfile.write(s.GetString(), strlen(s.GetString()));
      outfile.close();
   }
}

void JobShopInstance::readFromFile(const std::string& fileName)
{
   std::ifstream inFile;

   *this = JobShopInstance();
   
   inFile.open(fileName, std::ios::in);
   if (inFile.is_open()) {
      std::stringstream buffer;
      buffer << inFile.rdbuf();
      std::string json = buffer.str();

      rapidjson::Document doc;
      doc.Parse(json.c_str());

      numJobs = doc["jobs-count"].GetInt();
      numMachines = doc["machines-count"].GetInt();
      distType = static_cast<DistType>(doc["distance-type"].GetInt());
      theta = doc["theta"].GetDouble();

      rapidjson::Value& jobsValue = doc["jobs"];
      auto jobsArray = jobsValue.GetArray();
      for (auto& job : jobsArray) {
         jobs.emplace_back();
         rapidjson::Value& operationValue = job["operations"];
         auto operations = operationValue.GetArray();
         for (auto& operation : operations) {
            int    jobId     = operation["job"].GetInt();
            int    machine   = operation["machine"].GetInt();
            double mean      = operation["mean"].GetDouble();
            double variance  = operation["variance"].GetDouble();
            int    jobSeqIdx = operation["jobSeqIdx"].GetInt();
            Operation op(jobId, machine, mean, variance, jobSeqIdx);
            jobs.back().push_back(op);
         }
      }

      rapidjson::Value& duedatesValue = doc["duedates"];
      auto duedatesArray = duedatesValue.GetArray();
      for (auto& duedate : duedatesArray) {
         dueDates.push_back(duedate.GetDouble());
      }

      rapidjson::Value& weightsValue = doc["job-weights"];
      auto weightsArray = weightsValue.GetArray();
      for (auto& weight : weightsArray) {
         jobWeights.push_back(weight.GetInt());
      }
   }
   buildOperationsPerMachine();
}

void JobShopInstance::generateRandomInstance(unsigned seed) {
    rng.seed(seed);
    generateRandomRoutes();
    generateRandomProcessingMeans();
    //generateRealisticDueDates();
    generateDueDates();

    for (int j = 0; j < numJobs; ++j) {
        jobWeights[j] = randInt(1, 10);
    }
}

double JobShopInstance::sampleProcessingTime(const Operation& op) const {
    switch (distType) {
    case MEAN:
       return std::max(0.1, op.mean);
    case NORMAL:
       return std::max(0.1, normalSample(op.mean, theta * op.mean));
    case UNIFORM:
        return std::max(0.1, uniformSample(op.mean - theta * op.mean,
            op.mean + theta * op.mean));
    case EXPONENTIAL:
        return exponentialSample(1.0 / op.mean);
    default:
        return op.mean;
    }
}

double JobShopInstance::getMeanProcessingTime(const Operation& op) const {
    return op.mean;
}