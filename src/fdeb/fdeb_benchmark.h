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
 * Author: Yifan Sun (yifansun@coe.neu.edu)
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

#ifndef SRC_FDEB_FDEB_BENCHMARK_H_
#define SRC_FDEB_FDEB_BENCHMARK_H_

#include <vector>
#include <string>

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class FdebBenchmark : public Benchmark {
 protected:
  std::string data_name_;
 
  int node_count_;
  int edge_count_;
  std::vector<float> node_x_;
  std::vector<float> node_y_;
  std::vector<int> edge_src_;
  std::vector<int> edge_dst_;

  std::vector<float> compatibility_;
  std::vector<std::vector<float>> point_x_;
  std::vector<std::vector<float>> point_y_;

  int num_cycles_ = 6;
  int init_iter_count_ = 50;
  float init_step_size_ = 0.04;

  void LoadNodeData(const std::string &file_name);
  void LoadEdgeData(const std::string &file_name);
  void PrintNodes();
  void PrintEdges();
  void PrintSubdevisedEdges();

  void FdebCpu();
  void CalculateCompatibility();
  void BundlingCpu();
  void BundlingIterCpu(float step);
  void InitSubdivisionPoint();
  int GenerateSubdivisionPoint(int num_point);

 public:
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  void SetDataName(std::string data_name) {
    data_name_ = data_name;
  }
};

#endif  // SRC_FDEB_FDEB_BENCHMARK_H_
