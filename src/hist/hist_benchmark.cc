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

#include "src/hist/hist_benchmark.h"
#include <cstdio>
#include <cstdlib>

void HistBenchmark::Initialize() {
  pixels_ = new uint32_t[num_pixel_];
  unsigned int seed = time(NULL);
  for (uint32_t i = 0; i < num_pixel_; i++) {
    pixels_[i] = rand_r(&seed) % num_color_;
  }

  histogram_ = new uint32_t[num_color_]();
}

void HistBenchmark::Verify() {
  uint32_t *cpu_histogram = new uint32_t[num_color_]();
  for (uint32_t i = 0; i < num_pixel_; i++) {
    cpu_histogram[pixels_[i]]++;
  }

  bool has_error = false;
  for (uint32_t i = 0; i < num_color_; i++) {
    if (cpu_histogram[i] != histogram_[i]) {
      printf("At color %d, expected to be %d, but was %d\n", i,
             cpu_histogram[i], histogram_[i]);
      has_error = true;
      exit(-1);
    }
  }

  if (!has_error) {
    printf("Passed.\n");
  }
}

void HistBenchmark::Summarize() {
  printf("Image: \n");
  for (uint32_t i = 0; i < num_pixel_; i++) {
    printf("%d ", pixels_[i]);
  }
  printf("\n");

  printf("Histogram: \n");
  for (uint32_t i = 0; i < num_color_; i++) {
    printf("%d ", histogram_[i]);
  }
  printf("\n");
}

void HistBenchmark::Cleanup() {
  delete pixels_;
  delete histogram_;
}
