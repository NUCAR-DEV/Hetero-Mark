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
 * Author: Shi Dong (shidong@coe.neu.edu)
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

#include "src/knn/hc/knn_hc_benchmark.h"

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

void KnnHcBenchmark::Initialize() {
  KnnBenchmark::Initialize();

  h_locations_ = new LatLong[num_records_];
  h_distances_ = new float[num_records_];
  for (int i = 0; i < num_records_; i++) {
    h_locations_[i].lat = locations_.at(i).lat;
    h_locations_[i].lng = locations_.at(i).lng;
  }
}

void KnnHcBenchmark::Run() {
  int num_gpu_records = partitioning_ * num_records_;
  int num_cpu_records = (1 - partitioning_) * num_records_;
  printf("Num gpu records is %d \n", num_gpu_records);
  printf("Num cpu records is %d \n", num_cpu_records);

  std::atomic_int cpu_worklist;
  std::atomic_int gpu_worklist;
  gpu_worklist.store(0);
  cpu_worklist.store(partitioning_ * num_records_);

  // float *d_distances = new float[num_records_];
  hc::array<float, 1> av_distance(num_records_);
  hc::array<LatLong, 1> av_loc(num_records_);
  hc::copy(h_locations_, av_loc);
  // LatLong *latLong = h_locations_;
  float lat = latitude_;
  float lng = longitude_;

  hc::extent<1> kernel_ext(256);
  auto fut = parallel_for_each(kernel_ext, [&](hc::index<1> i)[[hc]] {
    int tid;
    tid = gpu_worklist.fetch_add(1, std::memory_order_seq_cst);
    while (true) {
      if (tid >= num_gpu_records) {
        break;
      }
      av_distance[tid] = static_cast<float>(
          sqrt((lat - av_loc[tid].lat) * (lat - av_loc[tid].lat) +
               (lng - av_loc[tid].lng) * (lng - av_loc[tid].lng)));

      tid = gpu_worklist.fetch_add(1, std::memory_order_seq_cst);
    }

    tid = cpu_worklist.fetch_add(1, std::memory_order_seq_cst);
    while (true) {
      if (tid >= num_gpu_records) {
        break;
      }
      av_distance[tid] = static_cast<float>(
          sqrt((lat - av_loc[tid].lat) * (lat - av_loc[tid].lat) +
               (lng - av_loc[tid].lng) * (lng - av_loc[tid].lng)));

      tid = cpu_worklist.fetch_add(1, std::memory_order_seq_cst);
    }
  });
  KnnCPU(h_locations_, h_distances_, num_records_, num_gpu_records, latitude_,
         longitude_, &cpu_worklist, &gpu_worklist);
  fut.wait();
  printf("%d\n", gpu_worklist.load(std::memory_order_seq_cst));

  // find the results Count least distances
  findLowest(records_, h_distances_, num_records_, k_value_);

  for (int i = 0; i < k_value_; i++) {
    printf("%s --> Distance=%f\n", records_[i].recString, records_[i].distance);
  }
}

void KnnHcBenchmark::Cleanup() { KnnBenchmark::Cleanup(); }
