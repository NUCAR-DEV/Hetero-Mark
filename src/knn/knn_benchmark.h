/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#ifndef SRC_KNN_KNN_BENCHMARK_H_
#define SRC_KNN_KNN_BENCHMARK_H_

#include <atomic>
#include <string>
#include <vector>

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class LatLong {
 public:
  float lat;
  float lng;
};

class Record {
 public:
  char recString[53];
  float distance;
};

class KnnBenchmark : public Benchmark {
 protected:
  /**
   * The CPU code for running KNN
   */
  std::vector<Record> records_;
  std::vector<LatLong> locations_;
  std::atomic_int *worklist_;
  std::atomic_int *gpu_worklist_;
  std::atomic_int *cpu_worklist_;
  LatLong *h_locations_ = nullptr;
  float *h_distances_ = nullptr;
  std::string filename_ = "";
  double latitude_ = 0.0;
  double longitude_ = 0.0;
  int num_records_ = 0;
  int k_value_ = 10;
  double partitioning_ = 0.6;
  void KnnCPU(LatLong *h_locations, float *h_distances, int num_records,
              int num_gpu_records, float lat, float lng,
              std::atomic_int *cpu_worklist, std::atomic_int *gpu_worklist);
  int loadData(std::string filename, std::vector<Record> *records,
               std::vector<LatLong> *locations);
  void findLowest(std::vector<Record> *records, float *distances,
                  int numRecords, int topN);
  float *output_distances_ = nullptr;

 public:
  KnnBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override = 0;
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void setFilename(std::string filename) { filename_ = filename; }
  void setLatitude(double latitude) { latitude_ = latitude; }
  void setLongitude(double longitude) { longitude_ = longitude; }
  void setKValue(int k_value) { k_value_ = k_value; }
};

#endif  // SRC_KNN_KNN_BENCHMARK_H_
