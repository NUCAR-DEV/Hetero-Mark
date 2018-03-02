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
#include <hsa/hsa.h>

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
      BundlingIterGpuCollaborative();
    }

    step_size_ = step_size_ / 2.0;
    iter = iter * 2 / 3;
  }

  SaveSubdevisedEdges("out_gpu.data");

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

void FdebHcBenchmark::BundlingIterGpuCollaborative() {
  int col = col_;
  int edge_count = edge_count_;
  float kp = kp_;
  // int group_size = 64;
  hc::extent<1> ext(edge_count_ * col_);

  std::atomic_int *signals = new std::atomic_int[edge_count * col];
  for (int i = 0; i < edge_count; i++) {
    for (int j = 0; j < col; j++) {
      if (j == 0 || j == col - 1) {
        signals[i * col + j] = 2;
      } else {
        signals[i * col + j] = 0;
      }
    }
  }

  hc::array<float, 1> d_comp(edge_count * edge_count);
  hc::array<float, 1> d_point_x(ext);
  hc::array<float, 1> d_point_y(ext);

  hc::copy(compatibility_.data(), d_comp);
  hc::copy(point_x_.data(), d_point_x);
  hc::copy(point_y_.data(), d_point_y);

  hsa_status_t err;
  err = hsa_memory_register(force_x_.data(), edge_count * col * sizeof(float));
  if (err != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to map memory\n");
    exit(-1);
  }

  err = hsa_memory_register(force_y_.data(), edge_count * col * sizeof(float));
  if (err != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to map memory\n");
    exit(-1);
  }

  float *a_force_x = force_x_.data();
  float *a_force_y = force_y_.data();
  // float *d_point_x = point_x_.data();
  // float *d_point_y = point_y_.data();
  // hc::array_view<float, 1> a_force_x(edge_count * col);
  // hc::array_view<float, 1> a_force_y(edge_count * col);

  std::thread cpu_thread([&] {
    while (true) {
      bool finished = true;
      // printf("edge_count %d, col %d\n", edge_count, col);
      for (int i = 0; i < edge_count * col; i++) {
        // printf("signals[%d] = %d.\n", i, signals[i].load(std::memory_order_relaxed));
        if (signals[i] == 1) {
          // float force_x = a_force_x[i].load(std::memory_order_seq_cst);
          // float force_y = a_force_y[i].load(std::memory_order_seq_cst);
          float force_x = force_x_[i];
          float force_y = force_y_[i];
          point_x_[i] += step_size_ * force_x;
          point_y_[i] += step_size_ * force_y;
          // signals[i] = 2;
          signals[i].fetch_add(1, std::memory_order_seq_cst);
          // printf("Moving point %d %f, %f, %f, %f\n", i,
          //     force_x, force_y, point_x_[i], point_y_[i]);
        } else if (signals[i] == 0) {
          finished = false;
        }

      }
      if (finished) {
          return;
      }

      // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  });

  auto fut = hc::parallel_for_each(ext, [&](hc::index<1> idx) [[hc]] {
    int point_id = idx[0];
    int i = point_id / col;
    int k = point_id % col;

    if (point_id >= edge_count * col) return;
    if (k == 0 || k == col - 1) return;

    float force_x = 0;
    float force_y = 0;

    for (int j = 0; j < edge_count; j++) {
      if (j == i) continue;
      float x1 = d_point_x[i * col + k];
      float y1 = d_point_y[i * col + k];
      float x2 = d_point_x[j * col + k];
      float y2 = d_point_y[j * col + k];
      float compatibility = d_comp[i * edge_count + j];
      float dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

      if (dist > 0 && compatibility > 0) {
        float x = x2 - x1;
        float y = y2 - y1;

        force_x += x / dist * compatibility;
        force_y += y / dist * compatibility;
      }
    }

    // Self force
    float x = d_point_x[i * col + k];
    float y = d_point_y[i * col + k];
    float x_p = d_point_x[i * col + k - 1];
    float y_p = d_point_y[i * col + k - 1];
    float x_n = d_point_x[i * col + k + 1];
    float y_n = d_point_y[i * col + k + 1];

    force_x += kp * (x_p - x);
    force_y += kp * (y_p - y);
    force_x += kp * (x_n - x);
    force_y += kp * (y_n - y);

    // Normalize
    float mag = sqrt(force_x * force_x + force_y * force_y);
    if (mag > 0) {
      force_x /= mag;
      force_y /= mag;
    }

    // a_force_x[i * col + k].store(force_x, std::memory_order_relaxed);
    // a_force_y[i * col + k].store(force_y, std::memory_order_relaxed);
    // a_force_x[point_id] = force_x;
    // a_force_y[point_id] = force_y;
    a_force_x[point_id] = force_x;
    a_force_y[point_id] = force_y;

    // Signal
    signals[i*col + k].fetch_add(1, std::memory_order_seq_cst);
  });

  

  fut.wait();

  
  cpu_thread.join();
  
  // MovePointsCpu();

  delete[] signals;
  // delete[] a_force_x;
  // delete[] a_force_y;

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
