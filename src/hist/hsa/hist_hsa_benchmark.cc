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

#include "src/hist/hsa/hist_hsa_benchmark.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "src/hist/hsa/kernels.h"

void HistHsaBenchmark::Initialize() {
  HistBenchmark::Initialize();
  HIST_init(0);
}

void HistHsaBenchmark::Run() {
  uint32_t *pixels = reinterpret_cast<uint32_t *>(
      malloc_global(num_pixel_ * sizeof(uint32_t)));
  uint32_t *histogram = reinterpret_cast<uint32_t *>(
      malloc_global(num_color_ * sizeof(uint32_t)));

  memcpy(pixels, pixels_, num_pixel_ * sizeof(uint32_t));

  SNK_INIT_LPARM(lparm, 0);
  lparm->ndim = 1;
  lparm->gdims[0] = 1024;
  lparm->ldims[0] = 64;
  HIST(pixels, histogram, num_color_, num_pixel_, lparm);

  for (int i = 0; i < 256; i++) {
    printf("%d ", histogram[i]);
  }

  memcpy(histogram_, histogram, num_color_ * sizeof(uint32_t));

  free_global(pixels);
  free_global(histogram);
}

void HistHsaBenchmark::Cleanup() { HistBenchmark::Cleanup(); }
