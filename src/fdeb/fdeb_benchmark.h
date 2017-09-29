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
  std::string input_file_;
 
  int edge_count_;

  std::vector<float> edge_src_x_;
  std::vector<float> edge_src_y_;
  std::vector<float> edge_dst_x_;
  std::vector<float> edge_dst_y_;

  std::vector<std::vector<float>> compatibility_;
  std::vector<std::vector<float>> point_x_;
  std::vector<std::vector<float>> point_y_;
  std::vector<std::vector<float>> force_x_;
  std::vector<std::vector<float>> force_y_;

  int num_cycles_ = 6;
  int init_iter_count_ = 50;
  float init_step_size_ = 0.04;

  void LoadNodeData(const std::string &file_name);
  void LoadEdgeData(const std::string &file_name);
  void PrintEdges();
  void PrintSubdevisedEdges();
  void SaveSubdevisedEdges(const std::string &filename);

  void FdebCpu();
  void CalculateCompatibility();
  void BundlingCpu();
  void InitSubdivisionPoint();
  void InitForce(int num_point_);
  int GenerateSubdivisionPoint(int num_point);
  void BundlingIterCpu(int num_point, float step);
  void UpdateForceCpu(int num_point);
  void MovePointsCpu(int num_point, float step);

 public:
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  void SetInputFile(const std::string &input_file) {
    input_file_ = input_file;
  }
};

#endif  // SRC_FDEB_FDEB_BENCHMARK_H_
