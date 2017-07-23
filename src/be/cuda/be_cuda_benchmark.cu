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

#include <cstdio>
#include <cstdlib>
#include "src/be/cuda/be_cuda_benchmark.h"
#include "src/common/time_measurement/time_measurement_impl.h"

void BeCudaBenchmark::Initialize() {
  BeBenchmark::Initialize();
  timer_ = new TimeMeasurementImpl();

  cudaMalloc(&d_bg_, width_ * height_ * channel_ * sizeof(float));
  cudaMalloc(&d_fg_, width_ * height_ * channel_ * sizeof(uint8_t));
}

void BeCudaBenchmark::Run() {
  if (collaborative_execution_) {
    CollaborativeRun();
  } else {
    NormalRun();
  }
}

__global__ void BackgroundExtraction(uint8_t *frame, float *bg, uint8_t *fg,
                                     uint32_t width, uint32_t height,
                                     uint32_t channel, uint8_t threshold,
                                     float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > width * height * channel) {
    return;
  }

  uint8_t diff;
  if (frame[tid] > bg[tid]) {
    diff = frame[tid] - bg[tid];
  } else {
    diff = bg[tid] - frame[tid];
  }

  if (diff > threshold) {
    fg[tid] = frame[tid];
  } else {
    fg[tid] = 0;
  }

  bg[tid] = bg[tid] * (1 - alpha) + frame[tid] * alpha;
}

void BeCudaBenchmark::CollaborativeRun() { printf("Collaborative run\n"); }

void BeCudaBenchmark::NormalRun() {
  printf("Normal run\n");
  int num_pixels = width_ * height_;
  uint8_t *frame;

  // Reset background image
  video_.open(input_file_);
  frame = nextFrame();
  float *temp_bg = new float[num_pixels * channel_];
  for (uint32_t i = 0; i < num_pixels * channel_; i++) {
    temp_bg[i] = static_cast<float>(frame[i]);
  }
  cudaMemcpy(d_bg_, temp_bg, num_pixels * channel_ * sizeof(float),
             cudaMemcpyHostToDevice);

  uint8_t *d_frame;
  cudaMalloc(&d_frame, num_pixels * channel_ * sizeof(uint8_t));

  while (true) {
    frame = nextFrame();
    if (frame == NULL) {
      break;
    }

    dim3 block_size(64);
    dim3 grid_size((num_pixels * channel_ + block_size.x - 1) / block_size.x);

    cudaMemcpy(d_frame, frame, num_pixels * channel_ * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    BackgroundExtraction<<<grid_size, block_size>>>(
        d_frame, d_bg_, d_fg_, width_, height_, channel_, threshold_, alpha_);

    cudaMemcpy(foreground_.data(), d_fg_,
               num_pixels * channel_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (generate_output_) {
      cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3,
                           foreground_.data(), cv::Mat::AUTO_STEP);
      video_writer_ << output_frame;
    }
  }
}

void BeCudaBenchmark::Summarize() {
  timer_->Summarize();
  BeBenchmark::Summarize();
}

void BeCudaBenchmark::Cleanup() {
  delete timer_;
  cudaFree(d_bg_);
  cudaFree(d_fg_);
  BeBenchmark::Cleanup();
}
