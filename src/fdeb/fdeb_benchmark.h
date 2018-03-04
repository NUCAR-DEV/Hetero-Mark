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

#include <string>
#include <vector>

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class FdebBenchmark : public Benchmark {
 protected:
  bool collaborative_;
  int gpu_batch_;    // If collaborative mode, the number of points that the
                     // GPU process in one kernel
  bool use_atomic_;  // If set, the GPU implement will use atomic operations.

  std::string input_file_;

  int num_cycles_;
  int init_iter_count_;
  float init_step_size_;
  float kp_;

  int edge_count_;
  int num_subpoint_;
  int col_;  // Number of all points on an edge.
  float step_size_;

  std::vector<float> edge_src_x_;
  std::vector<float> edge_src_y_;
  std::vector<float> edge_dst_x_;
  std::vector<float> edge_dst_y_;

  std::vector<float> compatibility_;
  std::vector<float> point_x_;
  std::vector<float> point_y_;
  std::vector<float> force_x_;
  std::vector<float> force_y_;

  void LoadNodeData(const std::string &file_name);
  void LoadEdgeData(const std::string &file_name);
  void PrintEdges();
  void PrintSubdevisedEdges();
  void SaveSubdevisedEdges(const std::string &filename);

  void FdebCpu();
  void CalculateCompatibility();
  float AngleCompatibility(int i, int j);
  float ScaleCompatibility(int i, int j);
  float PositionCompatibility(int i, int j);
  float VisibilityCompatibility(int i, int j);
  void BundlingCpu();
  void InitSubdivisionPoint();
  void InitForce();

  void GenerateSubdivisionPoint();
  void BundlingIterCpu();
  void UpdateForceCpu();
  void MovePointsCpu();

 public:
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  void SetInputFile(const std::string &input_file) { input_file_ = input_file; }

  void SetNumCycle(int cycle) { num_cycles_ = cycle; }

  void SetInitStepSize(float step_size) { init_step_size_ = step_size; }

  void SetKp(float kp) { kp_ = kp; }

  void SetInitIter(float init_iter) { init_iter_count_ = init_iter; }

  void SetCollaborative(bool collaborative) { collaborative_ = collaborative; }

  void SetGpuBatch(int gpu_batch) { gpu_batch_ = gpu_batch; }

  void SetUseAtomic(bool atomic) { use_atomic_ = atomic; }
};

#endif  // SRC_FDEB_FDEB_BENCHMARK_H_
