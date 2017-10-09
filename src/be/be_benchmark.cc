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

#include "src/be/be_benchmark.h"
#include <cstdio>
#include <cstdlib>

void BeBenchmark::Initialize() {
  if (!video_.open(input_file_)) {
    fprintf(stderr, "Fail to open input file.\n");
    exit(1);
  }

  width_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_HEIGHT));
  channel_ = 3;
  num_frames_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_COUNT));
  if (max_frames_ != 0 && max_frames_ < num_frames_) {
    num_frames_ = max_frames_;
  }

  if (generate_output_) {
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    video_writer_.open("gpu_output.avi", codec, video_.get(CV_CAP_PROP_FPS),
                       cv::Size(width_, height_), true);
    cpu_video_writer_.open("cpu_output.avi", codec, video_.get(CV_CAP_PROP_FPS),
                           cv::Size(width_, height_), true);
  }

  background_.resize(width_ * height_ * channel_);
  foreground_.resize(width_ * height_ * channel_);
}

uint8_t *BeBenchmark::nextFrame() {
  cv::Mat image;
  cpu_gpu_logger_->CPUOn();
  if (video_.read(image)) {
    // printf("Image type %d, dims %d, rows %d, cols %d\n",
    //     image.type(), image.dims, image.rows, image.cols);
    uint8_t *buffer = new uint8_t[width_ * height_ * channel_];
    memcpy(buffer, image.data, width_ * height_ * channel_);
    cpu_gpu_logger_->CPUOff();
    return buffer;
  } else {
    cpu_gpu_logger_->CPUOff();
    return NULL;
  }
}

void BeBenchmark::Verify() {
  CpuRun();
  Match();
}

void BeBenchmark::CpuRun() {
  cv::Mat image;
  int num_pixels = width_ * height_ * channel_;
  cpu_foreground_.resize(num_pixels);

  // Reset background image
  video_.open(input_file_);
  video_ >> image;
  uint8_t *frame = image.ptr();
  for (int i = 0; i < num_pixels; i++) {
    background_[i] = static_cast<float>(frame[i]);
  }

  // Run on CPU
  uint64_t frame_count = 0;
  while (true) {
    if (frame_count >= num_frames_) {
      break;
    }

    if (!video_.read(image)) {
      break;
    }
    frame = image.ptr();
    frame_count++;

    for (int i = 0; i < num_pixels; i++) {
      uint8_t diff = 0;
      if (frame[i] > background_[i]) {
        diff = frame[i] - background_[i];
      } else {
        diff = -frame[i] + background_[i];
      }
      if (diff > threshold_) {
        cpu_foreground_[i] = frame[i];
      } else {
        cpu_foreground_[i] = 0;
      }
      background_[i] = background_[i] * (1 - alpha_) + frame[i] * alpha_;
    }

    if (generate_output_) {
      cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3,
                           cpu_foreground_.data(), cv::Mat::AUTO_STEP);
      cpu_video_writer_ << output_frame;
    }
  }
}

void BeBenchmark::Match() {
  uint32_t num_bytes = width_ * height_ * channel_;
  for (uint32_t i = 0; i < num_bytes; i++) {
    if (cpu_foreground_ != foreground_) {
      printf("Mismatch in byte %u\n", i);
      exit(-1);
    }
  }
  printf("Passed!\n");
}

void BeBenchmark::Summarize() {}

void BeBenchmark::Cleanup() {
  if (generate_output_) {
    cpu_video_writer_.release();
    video_writer_.release();
  }
}
