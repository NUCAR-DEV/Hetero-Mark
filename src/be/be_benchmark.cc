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

#include <cstdlib>
#include <cstdio>
#include "src/be/be_benchmark.h"

void BeBenchmark::Initialize() {
  if (!video_.open(input_file_)) {
    fprintf(stderr, "Fail to open input file.\n");
    exit(1);
  }

  width_= static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_HEIGHT));
  channel_ = 3;
  num_frames_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_COUNT));
  printf("width %d, height %d, channel %d, num_frames %d\n", 
      width_, height_, channel_, num_frames_);

  video_writer_.open("gpu_ouput.avi", 
      CV_FOURCC('M', 'P', 'E', 'G'),
      video_.get(CV_CAP_PROP_FPS), 
      cv::Size(width_, height_), 
      true);

  foreground_.resize((uint64_t)width_ * height_ * channel_ *  num_frames_);
  background_.resize(width_ * height_ * channel_);

  collaborative_execution_ = true;
}

uint8_t *BeBenchmark::nextFrame() {
  cv::Mat image;
  if (video_.read(image)) {
    printf("Image type %d, dims %d, rows %d, cols %d\n", 
        image.type(), image.dims, image.rows, image.cols);
    uint8_t *buffer = new uint8_t[width_ * height_ * channel_];
    memcpy(buffer, image.data, width_ * height_ * channel_);
    return buffer;
  } else {
    return NULL;
  }
}

void BeBenchmark::Verify() {
  int num_pixels = width_ * height_ * channel_;
  uint8_t *cpu_foreground = new uint8_t[num_frames_ * num_pixels];

  // Reset background image
  video_.open(input_file_);
  uint8_t *frame = nextFrame();
  for (int i = 0; i < num_pixels; i++) {
    background_[i] = static_cast<float>(frame[i]);
  }

  // Run on CPU
  uint64_t frame_count = 0;
  while(frame != NULL) {
    printf("Frame %lu\n", frame_count);
    for (int i = 0; i < num_pixels; i++) {
      uint8_t diff = 0;
      if (frame[i] > background_[i]) {
        diff = frame[i] - background_[i];
      } else {
        diff = -frame[i] + background_[i];
      } 
      if (diff > threshold_) {
        cpu_foreground[frame_count * num_pixels + i] = frame[i];
      }
      background_[i] = background_[i] * (1 - alpha_) + frame[i] * alpha_;
    }

    cv::Mat output_frame(cv::Size(width_, height_), CV_8UC3, 
        cpu_foreground + frame_count * num_pixels,
        cv::Mat::AUTO_STEP);
    video_writer_ << output_frame;

    frame_count++;
    delete[] frame;
    frame = nextFrame();
  }

  // Match
  bool has_error = false;
  for (uint64_t i = 0; i < num_frames_; i++) {
    for (uint64_t j = 0; j < num_pixels; j++) {
      uint64_t id = i * num_pixels + j;
      if (foreground_[id] != cpu_foreground[id]) {
        printf("Frame %lu, pixel %lu, expected %d, but was %d\n", i, j,
          cpu_foreground[id], foreground_[id]);
        has_error = true;
        return;
      }
    }
  }
  if (!has_error) {
    printf("Passed!\n");
  }

  delete[] cpu_foreground;
}

void BeBenchmark::Summarize() {}

void BeBenchmark::Cleanup() {
}
