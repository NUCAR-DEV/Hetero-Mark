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
#include "src/fdeb/hc/fdeb_hc_benchmark.h"

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

void FdebHcBenchmark::Initialize() {
  FdebBenchmark::Initialize();
}

void FdebHcBenchmark::Run() {
  if (collaborative_) {
    CollaborativeRun();
  } else {
    NormalRun();
  }
}

void FdebHcBenchmark::CollaborativeRun() {
}

void FdebHcBenchmark::NormalRun() {
  CalculateCompatibility();

  num_subpoint_ = 0;
  int iter = init_iter_count_;
  step_size_ = init_step_size_;

  InitSubdivisionPoint();
  for (int i = 0; i < num_cycles_; i++) {
    printf("GPU Cycle %d\n", i);

    GenerateSubdivisionPoint();
    InitForce();

    for (int j = 0; j < iter; j++) {
      printf("\tGpu Iter %d\n", j);
      BundlingIterGpu();
    }

    step_size_ = step_size_ / 2.0;
    iter = iter * 2 / 3;
  }

  SaveSubdevisedEdges("out_gpu.data");
}

void FdebHcBenchmark::BundlingIterGpu() {
  UpdateForceGpu();
  MovePointsCpu();
}

void FdebHcBenchmark::UpdateForceGpu() {
  hc::extent<1> ext(edge_count_ * col_); // Extra wi for end point

  hc::array_view<float, 1> av_comp(hc::extent<1>(edge_count_*edge_count_),
      compatibility_);
  hc::array_view<float, 1> av_point_x(ext, point_x_);
  hc::array_view<float, 1> av_point_y(ext, point_y_);
  hc::array_view<float, 1> av_force_x(ext, force_x_);
  hc::array_view<float, 1> av_force_y(ext, force_y_);

  av_force_x.discard_data();
  av_force_y.discard_data();

  int edge_count = edge_count_;
  int col = col_;
  float kp = kp_;
  hc::extent<2> kernel_ext(edge_count_, col_); // Extra wi for end point
  hc::parallel_for_each(kernel_ext, [=](hc::index<2> idx) [[hc]] {
      int i = idx[0];
      int k = idx[1];

      if (k == 0 || k == col - 1) return;

      float force_x = 0;
      float force_y = 0;

      for (int j = 0; j < edge_count; j++) {
        if (j == i) continue;
        float x1 = av_point_x[i * col + k];
        float y1 = av_point_y[i * col + k];
        float x2 = av_point_x[j * col + k];
        float y2 = av_point_y[j * col + k];
        float compatibility = av_comp[i * edge_count + j];
        float dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

        if (dist > 0) {
          float x = x2 - x1;
          float y = y2 - y1;

          force_x += x / dist * compatibility;
          force_y += y / dist * compatibility;
        }
      }

      // Self force
      float x = av_point_x[i * col + k];
      float y = av_point_y[i * col + k];
      float x_p = av_point_x[i * col + k - 1];
      float y_p = av_point_y[i * col + k - 1];
      float x_n = av_point_x[i * col + k + 1];
      float y_n = av_point_y[i * col + k + 1];

      force_x += kp * (x_p - x);
      force_y += kp * (y_p - y);
      force_x += kp * (x_n - x);
      force_y += kp * (y_n - y);

      // Normalize
      float mag = sqrt(force_x * force_x + force_y * force_y);
      force_x /= mag;
      force_y /= mag;

      av_force_x[i * col + k] = force_x;
      av_force_y[i * col + k] = force_y;
  });

  av_force_x.synchronize();
  av_force_y.synchronize();
}

void FdebHcBenchmark::Cleanup() {
  FdebBenchmark::Cleanup();
}
