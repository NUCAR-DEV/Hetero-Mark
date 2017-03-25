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
  num_frames_ = static_cast<uint32_t>(video_.get(CV_CAP_PROP_FRAME_COUNT));
}

uint8_t *BeBenchmark::nextFrame() {
  cv::Mat image;
  video_ >> image;
  uint8_t *buffer = new uint8_t[width_ * height_];
  return buffer;
}

void BeBenchmark::Verify() {
  /*
  uint8_t *cpu_foreground = new uint8_t[num_frames_ * num_pixels_];

  // Reset background image
  for (uint32_t i = 0; i < num_pixels_; i++) {
    background_[i] = data_[i];
  }

  // Run on CPU
  for (uint32_t i = 0; i < num_frames_; i++) {
    for (uint32_t j = 0; j < num_pixels_; j++) {
      uint32_t id = i * num_pixels_ + j;
      if (data_[id] > background_[j]) {
        cpu_foreground[id] = data_[id] - background_[j];
      } else {
        cpu_foreground[id] = background_[j] - data_[id];
      }
      background_[j] = background_[j] * (1 - alpha_) + data_[id] * alpha_;
    }
  }

  // Match
  bool has_error = false;
  for (uint32_t i = 0; i < num_frames_; i++) {
    for (uint32_t j = 0; j < num_pixels_; j++) {
      uint32_t id = i * num_pixels_ + j;
      if (foreground_[id] != cpu_foreground[id]) {
        printf("Frame %d, pixel %d, expected %d, but was %d\n", i, j,
          cpu_foreground[id], foreground_[id]);
        has_error = true;
      }
    }
  }
  if (!has_error) {
    printf("Passed!\n");
  }
  */
}

void BeBenchmark::Summarize() {}

void BeBenchmark::Cleanup() {
}
