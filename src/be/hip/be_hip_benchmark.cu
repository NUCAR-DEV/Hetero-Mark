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

#include "src/be/hip/be_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#include "src/common/time_measurement/time_measurement_impl.h"

void BeHipBenchmark::Initialize() {
  BeBenchmark::Initialize();
  timer_ = new TimeMeasurementImpl();

  hipMalloc(&d_bg_, width_ * height_ * channel_ * sizeof(float));
  hipMalloc(&d_fg_, width_ * height_ * channel_ * sizeof(uint8_t));
  hipMalloc(&d_frame_, width_ * height_ * channel_ * sizeof(uint8_t));
}

void BeHipBenchmark::Run() {
  if (collaborative_execution_) {
    CollaborativeRun();
  } else {
    NormalRun();
  }
  cpu_gpu_logger_->Summarize();
}

__global__ void BackgroundExtraction(hipLaunchParm lp, uint8_t *frame,
                                     float *bg, uint8_t *fg, uint32_t width,
                                     uint32_t height, uint32_t channel,
                                     uint8_t threshold, float alpha) {
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

void BeHipBenchmark::CollaborativeRun() {
  printf("Collaborative run\n");
  uint32_t num_pixels = width_ * height_;
  uint8_t *frame = NULL;

  std::thread gpuThread(&BeHipBenchmark::GPUThread, this);

  // Initialize background
  frame = nextFrame();
  float *temp_bg = new float[num_pixels * channel_];
  for (uint32_t i = 0; i < num_pixels * channel_; i++) {
    temp_bg[i] = static_cast<float>(frame[i]);
  }
  hipMemcpy(d_bg_, temp_bg, num_pixels * channel_ * sizeof(float),
            hipMemcpyHostToDevice);
  free(temp_bg);

  uint32_t frame_count = 0;
  while (true) {
    if (frame_count >= num_frames_) {
      break;
    }

    frame = nextFrame();
    if (!frame) {
      break;
    }
    frame_count++;

    std::lock_guard<std::mutex> lk(queue_mutex_);
    frame_queue_.push(frame);
    queue_condition_variable_.notify_all();
  }

  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    finished_ = true;
  }
  queue_condition_variable_.notify_all();

  gpuThread.join();
}

void BeHipBenchmark::GPUThread() {
  while (true) {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    queue_condition_variable_.wait(
        lk, [this] { return finished_ || !frame_queue_.empty(); });
    while (!frame_queue_.empty()) {
      ExtractAndEncode(frame_queue_.front());
      frame_queue_.pop();
    }
    if (finished_) break;
  }
}

void BeHipBenchmark::ExtractAndEncode(uint8_t *frame) {
  uint32_t num_pixels = width_ * height_;

  hipMemcpy(d_frame_, frame, num_pixels * channel_ * sizeof(uint8_t),
            hipMemcpyHostToDevice);

  dim3 block_size(64);
  dim3 grid_size((num_pixels * channel_ + block_size.x - 1) / block_size.x);

  cpu_gpu_logger_->GPUOn();
  hipLaunchKernel(HIP_KERNEL_NAME(BackgroundExtraction), dim3(grid_size),
                  dim3(block_size), 0, 0, d_frame_, d_bg_, d_fg_, width_,
                  height_, channel_, threshold_, alpha_);

  hipMemcpy(foreground_.data(), d_fg_, num_pixels * channel_ * sizeof(uint8_t),
            hipMemcpyDeviceToHost);
  cpu_gpu_logger_->GPUOff();
  if (generate_output_) {
    cpu_gpu_logger_->CPUOn();
    cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3, foreground_.data(),
                         cv::Mat::AUTO_STEP);
    video_writer_ << output_frame;
    cpu_gpu_logger_->CPUOff();
  }
}

void BeHipBenchmark::NormalRun() {
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
  hipMemcpy(d_bg_, temp_bg, num_pixels * channel_ * sizeof(float),
            hipMemcpyHostToDevice);
  free(temp_bg);

  uint8_t *d_frame;
  hipMalloc(&d_frame, num_pixels * channel_ * sizeof(uint8_t));

  uint32_t frame_count = 0;
  while (true) {
    if (frame_count >= num_frames_) {
      break;
    }

    frame = nextFrame();
    if (frame == NULL) {
      break;
    }

    dim3 block_size(64);
    dim3 grid_size((num_pixels * channel_ + block_size.x - 1) / block_size.x);

    hipMemcpy(d_frame, frame, num_pixels * channel_ * sizeof(uint8_t),
              hipMemcpyHostToDevice);

    cpu_gpu_logger_->GPUOn();
    hipLaunchKernel(HIP_KERNEL_NAME(BackgroundExtraction), dim3(grid_size),
                    dim3(block_size), 0, 0, d_frame, d_bg_, d_fg_, width_,
                    height_, channel_, threshold_, alpha_);

    hipMemcpy(foreground_.data(), d_fg_,
              num_pixels * channel_ * sizeof(uint8_t), hipMemcpyDeviceToHost);
    cpu_gpu_logger_->GPUOff();
    if (generate_output_) {
      cpu_gpu_logger_->CPUOn();
      cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3,
                           foreground_.data(), cv::Mat::AUTO_STEP);
      video_writer_ << output_frame;
      cpu_gpu_logger_->CPUOff();
    }

    delete[] frame;
    frame_count++;
  }

  hipFree(d_frame);
}

void BeHipBenchmark::Summarize() {
  timer_->Summarize();
  BeBenchmark::Summarize();
}

void BeHipBenchmark::Cleanup() {
  delete timer_;
  hipFree(d_bg_);
  hipFree(d_fg_);
  hipFree(d_frame_);
  BeBenchmark::Cleanup();
}
