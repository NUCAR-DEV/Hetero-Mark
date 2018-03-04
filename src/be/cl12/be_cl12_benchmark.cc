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

#include "src/be/cl12/be_cl12_benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "src/common/time_measurement/time_measurement_impl.h"

void BeCl12Benchmark::Initialize() {
  BeBenchmark::Initialize();

  ClBenchmark::InitializeCl();
  InitializeKernels();

  InitializeBuffers();
}

void BeCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);

  checkOpenCLErrors(err, "Failed to create program...\n");

  be_kernel_ = clCreateKernel(program_, "be_cl12", &err);
  checkOpenCLErrors(err, "Failed to create kernel BE\n");
}

void BeCl12Benchmark::InitializeBuffers() {
  // Create memory buffers for background, foreground and frames on the device
  cl_int err;
  d_bg_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     width_ * height_ * channel_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate background buffer");

  d_fg_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     width_ * height_ * channel_ * sizeof(uint8_t), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate foreground buffer");

  d_frame_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     width_ * height_ * channel_ * sizeof(uint8_t), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate frame buffer");
}

void BeCl12Benchmark::Run() {
  if (collaborative_execution_) {
    CollaborativeRun();
  } else {
    NormalRun();
  }

  cpu_gpu_logger_->Summarize();
}

void BeCl12Benchmark::NormalRun() {
  printf("Normal run\n");
  int num_pixels = width_ * height_;
  uint8_t *frame;

  cl_int ret;

  // Reset background image
  video_.open(input_file_);
  frame = nextFrame();
  float *temp_bg = new float[num_pixels * channel_];
  for (uint32_t i = 0; i < num_pixels * channel_; i++) {
    temp_bg[i] = static_cast<float>(frame[i]);
  }
  ret = clEnqueueWriteBuffer(cmd_queue_, d_bg_, CL_TRUE, 0,
                             num_pixels * channel_ * sizeof(float), temp_bg, 0,
                             NULL, NULL);
  checkOpenCLErrors(ret, "Copy bakcground\n");
  free(temp_bg);

  cl_mem d_frame;

  cl_int err;
  d_frame = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                           num_pixels * channel_ * sizeof(uint8_t), NULL, &err);

  checkOpenCLErrors(err, "Failed to allocate frame buffer");

  uint32_t frame_count = 0;
  while (true) {
    if (frame_count >= num_frames_) {
      break;
    }

    delete[] frame;
    frame = nextFrame();
    if (frame == NULL) {
      break;
    }

    frame_count++;

    // Set the arguments of the kernel
    ret = clSetKernelArg(be_kernel_, 0, sizeof(cl_mem),
                         reinterpret_cast<void *>(&d_frame));
    checkOpenCLErrors(ret, "Set kernel argument 0\n");

    ret = clSetKernelArg(be_kernel_, 1, sizeof(cl_mem),
                         reinterpret_cast<void *>(&d_bg_));
    checkOpenCLErrors(ret, "Set kernel argument 1\n");

    ret = clSetKernelArg(be_kernel_, 2, sizeof(cl_mem),
                         reinterpret_cast<void *>(&d_fg_));
    checkOpenCLErrors(ret, "Set kernel argument 2\n");

    ret = clSetKernelArg(be_kernel_, 3, sizeof(cl_uint),
                         reinterpret_cast<void *>(&width_));
    checkOpenCLErrors(ret, "Set kernel argument 3\n");

    ret = clSetKernelArg(be_kernel_, 4, sizeof(cl_uint),
                         reinterpret_cast<void *>(&height_));
    checkOpenCLErrors(ret, "Set kernel argument 4\n");

    ret = clSetKernelArg(be_kernel_, 5, sizeof(cl_uint),
                         reinterpret_cast<void *>(&channel_));
    checkOpenCLErrors(ret, "Set kernel argument 5\n");

    ret = clSetKernelArg(be_kernel_, 6, sizeof(cl_uint),
                         reinterpret_cast<void *>(&threshold_));
    checkOpenCLErrors(ret, "Set kernel argument 6\n");

    ret = clSetKernelArg(be_kernel_, 7, sizeof(float),
                         reinterpret_cast<void *>(&alpha_));
    checkOpenCLErrors(ret, "Set kernel argument 7\n");

    ret = clEnqueueWriteBuffer(cmd_queue_, d_frame, CL_TRUE, 0,
                               num_pixels * channel_ * sizeof(uint8_t), frame,
                               0, NULL, NULL);
    checkOpenCLErrors(ret, "Copy frame\n");

    size_t localThreads[1] = {64};
    size_t globalThreads[1] = {num_pixels * channel_};

    cpu_gpu_logger_->GPUOn();
    ret = clEnqueueNDRangeKernel(cmd_queue_, be_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(ret, "Enqueue ND Range.\n");
    clFinish(cmd_queue_);
    cpu_gpu_logger_->GPUOff();

    // Get data back
    ret = clEnqueueReadBuffer(cmd_queue_, d_fg_, CL_TRUE, 0,
                              num_pixels * channel_ * sizeof(uint8_t),
                              foreground_.data(), 0, NULL, NULL);
    checkOpenCLErrors(ret, "Copy data back\n");
    clFinish(cmd_queue_);

    if (generate_output_) {
      cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3,
                           foreground_.data(), cv::Mat::AUTO_STEP);
      video_writer_ << output_frame;
    }
  }
  ret = clReleaseMemObject(d_frame);
}

void BeCl12Benchmark::Cleanup() {
  cl_int ret;

  ret = clReleaseKernel(be_kernel_);
  ret = clReleaseProgram(program_);
  ret = clReleaseMemObject(d_bg_);
  ret = clReleaseMemObject(d_fg_);
  ret = clReleaseMemObject(d_frame_);

  checkOpenCLErrors(ret, "Release objects.\n");

  BeBenchmark::Cleanup();
}

void BeCl12Benchmark::Summarize() { BeBenchmark::Summarize(); }

void BeCl12Benchmark::ExtractAndEncode(uint8_t *frame) {
  cl_int ret;

  uint32_t num_pixels = width_ * height_;

  ret = clEnqueueWriteBuffer(cmd_queue_, d_frame_, CL_TRUE, 0,
                             num_pixels * channel_ * sizeof(uint8_t), frame, 0,
                             NULL, NULL);

  ret = clSetKernelArg(be_kernel_, 0, sizeof(cl_mem),
                       reinterpret_cast<void *>(&d_frame_));
  checkOpenCLErrors(ret, "Set kernel argument 0\n");

  ret = clSetKernelArg(be_kernel_, 1, sizeof(cl_mem),
                       reinterpret_cast<void *>(&d_bg_));
  checkOpenCLErrors(ret, "Set kernel argument 1\n");

  ret = clSetKernelArg(be_kernel_, 2, sizeof(cl_mem),
                       reinterpret_cast<void *>(&d_fg_));
  checkOpenCLErrors(ret, "Set kernel argument 2\n");

  ret = clSetKernelArg(be_kernel_, 3, sizeof(cl_uint),
                       reinterpret_cast<void *>(&width_));
  checkOpenCLErrors(ret, "Set kernel argument 3\n");

  ret = clSetKernelArg(be_kernel_, 4, sizeof(cl_uint),
                       reinterpret_cast<void *>(&height_));
  checkOpenCLErrors(ret, "Set kernel argument 4\n");

  ret = clSetKernelArg(be_kernel_, 5, sizeof(cl_uint),
                       reinterpret_cast<void *>(&channel_));
  checkOpenCLErrors(ret, "Set kernel argument 5\n");

  ret = clSetKernelArg(be_kernel_, 6, sizeof(cl_uint),
                       reinterpret_cast<void *>(&threshold_));
  checkOpenCLErrors(ret, "Set kernel argument 6\n");

  ret = clSetKernelArg(be_kernel_, 7, sizeof(cl_uint),
                       reinterpret_cast<void *>(&alpha_));
  checkOpenCLErrors(ret, "Set kernel argument 7\n");

  ret = clEnqueueWriteBuffer(cmd_queue_, d_frame_, CL_TRUE, 0,
                             num_pixels * channel_ * sizeof(uint8_t), frame, 0,
                             NULL, NULL);
  checkOpenCLErrors(ret, "Copy frame\n");

  size_t localThreads[1] = {64};
  size_t globalThreads[1] = {num_pixels * channel_};

  cpu_gpu_logger_->GPUOn();
  ret = clEnqueueNDRangeKernel(cmd_queue_, be_kernel_, CL_TRUE, NULL,
                               globalThreads, localThreads, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Enqueue ND Range.\n");
  clFinish(cmd_queue_);
  cpu_gpu_logger_->GPUOff();

  // Get data back
  ret = clEnqueueReadBuffer(cmd_queue_, d_fg_, CL_TRUE, 0,
                            num_pixels * channel_ * sizeof(uint8_t),
                            foreground_.data(), 0, NULL, NULL);
  checkOpenCLErrors(ret, "Copy data back\n");
  clFinish(cmd_queue_);

  delete[] frame;
  if (generate_output_) {
    cpu_gpu_logger_->CPUOn();
    cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3, foreground_.data(),
                         cv::Mat::AUTO_STEP);
    video_writer_ << output_frame;
    cpu_gpu_logger_->CPUOff();
  }
}

void BeCl12Benchmark::CollaborativeRun() {
  printf("Collaborative run\n");
  uint32_t num_pixels = width_ * height_;
  uint8_t *frame = NULL;
  cl_int ret;

  std::thread gpuThread(&BeCl12Benchmark::GPUThread, this);

  // Initialize background
  frame = nextFrame();
  float *temp_bg = new float[num_pixels * channel_];
  for (uint32_t i = 0; i < num_pixels * channel_; i++) {
    temp_bg[i] = static_cast<float>(frame[i]);
  }

  ret = clEnqueueWriteBuffer(cmd_queue_, d_bg_, CL_TRUE, 0,
                             num_pixels * channel_ * sizeof(float), temp_bg, 0,
                             NULL, NULL);
  checkOpenCLErrors(ret, "Copy bakcground\n");

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

    // printf("Frame started %d\n", frame_count);
    std::lock_guard<std::mutex> lk(queue_mutex_);
    frame_queue_.push(frame);
    queue_condition_variable_.notify_all();
  }

  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    // printf("Finished.\n");
    finished_ = true;
  }
  queue_condition_variable_.notify_all();

  gpuThread.join();
  // cudaStreamSynchronize(stream_);
}

void BeCl12Benchmark::GPUThread() {
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
