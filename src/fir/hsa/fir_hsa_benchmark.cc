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
 * Author: Yifan Sun (yifansun@coe.neu.edu)
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

#include "src/fir/hsa/fir_hsa_benchmark.h"
#include <cstdlib>
#include <cstdio>
#include "src/fir/hsa/kernels.h"

void FirHsaBenchmark::Initialize() {
  FirBenchmark::Initialize();

  // History saves data that carries to next kernel launch
  history_ = malloc_global(num_tap_);
  for (unsigned int i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }

  FIR_init(0);
}

void FirHsaBenchmark::Run() {
  for (unsigned int i = 0; i < num_block_; i++) {
    SNK_INIT_LPARM(lparm, 0);
    lparm->ndim = 1;
    lparm->gdims[0] = num_data_per_block_;
    lparm->ldims[0] = 64;
    FIR(input_ + i * num_data_per_block_, output_ + i * num_data_per_block_,
        coeff_, history_, num_tap_, lparm);
  }
}

void FirHsaBenchmark::Cleanup() {
  FirBenchmark::Cleanup();
  free_global(history_);
}
