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
#include <thread>
#include "src/be/cuda/be_cuda_benchmark.h"
#include "src/common/time_measurement/time_measurement_impl.h"

void BeCudaBenchmark::Initialize() {
  BeBenchmark::Initialize();
  timer_ = new TimeMeasurementImpl();

  cudaMalloc(&d_bg_, width_ * height_ * channel_ * sizeof(float));
  cudaMalloc(&d_fg_, width_ * height_ * channel_ * sizeof(uint8_t));
  cudaMalloc(&d_frame_, width_ * height_ * channel_ * sizeof(uint8_t));

  cudaStreamCreate(&stream_);
}

void BeCudaBenchmark::Run() {
  if (collaborative_execution_) {
    CollaborativeRun();
  } else {
    NormalRun();
  }
  cpu_gpu_logger_->Summarize();
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

void BeCudaBenchmark::CollaborativeRun() {
  printf("Collaborative run\n");
  uint32_t num_pixels = width_ * height_;
  uint8_t *frame = NULL;

  std::thread gpuThread(&BeCudaBenchmark::GPUThread, this);

  // Initialize background
  frame = nextFrame();
  float *temp_bg = new float[num_pixels * channel_];
  for (uint32_t i = 0; i < num_pixels * channel_; i++) {
    temp_bg[i] = static_cast<float>(frame[i]);
  }
  cudaMemcpy(d_bg_, temp_bg, num_pixels * channel_ * sizeof(float),
             cudaMemcpyHostToDevice);
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
  cudaStreamSynchronize(stream_);
}

void BeCudaBenchmark::GPUThread() {
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

void BeCudaBenchmark::ExtractAndEncode(uint8_t *frame) {
  uint32_t num_pixels = width_ * height_;

  cudaMemcpyAsync(d_frame_, frame, num_pixels * channel_ * sizeof(uint8_t),
                  cudaMemcpyHostToDevice, stream_);

  dim3 block_size(64);
  dim3 grid_size((num_pixels * channel_ + block_size.x - 1) / block_size.x);

  cpu_gpu_logger_->GPUOn();
  BackgroundExtraction<<<grid_size, block_size, 0, stream_>>>(
      d_frame_, d_bg_, d_fg_, width_, height_, channel_, threshold_, alpha_);
  cpu_gpu_logger_->GPUOff();

  cudaMemcpyAsync(foreground_.data(), d_fg_,
                  num_pixels * channel_ * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  delete[] frame;
  if (generate_output_) {
    cpu_gpu_logger_->CPUOn();
    cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3, foreground_.data(),
                         cv::Mat::AUTO_STEP);
    video_writer_ << output_frame;
    cpu_gpu_logger_->CPUOff();
  }
}

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
  free(temp_bg);

  uint8_t *d_frame;
  cudaMalloc(&d_frame, num_pixels * channel_ * sizeof(uint8_t));

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

    cudaMemcpy(d_frame, frame, num_pixels * channel_ * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    cpu_gpu_logger_->GPUOn();
    BackgroundExtraction<<<grid_size, block_size>>>(
        d_frame, d_bg_, d_fg_, width_, height_, channel_, threshold_, alpha_);
    cpu_gpu_logger_->GPUOff();

    cudaMemcpy(foreground_.data(), d_fg_,
               num_pixels * channel_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
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

  cudaFree(d_frame);
}

void BeCudaBenchmark::Summarize() {
  timer_->Summarize();
  BeBenchmark::Summarize();
}

void BeCudaBenchmark::Cleanup() {
  delete timer_;
  cudaFree(d_bg_);
  cudaFree(d_fg_);
  cudaFree(d_frame_);
  cudaStreamDestroy(stream_);
  BeBenchmark::Cleanup();
}
