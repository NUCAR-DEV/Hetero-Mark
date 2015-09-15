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

#include "src/hsa/iir_cl12_hsa/iir_benchmark.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <memory>

#include "src/hsa/iir_cl12_hsa/kernels.h"

void cpu_pariir(float *x, float *y, float *ns, float *dsec, float c, int len);

IirBenchmark::IirBenchmark(int len) {
  this->len = len;
}

void IirBenchmark::Cleanup() {
  delete[] X;
  delete[] gpu_Y;
  delete[] cpu_y;
  delete[] nsec;
  delete[] dsec;
}

void IirBenchmark::Initialize() {
  // Create the input and output arrays in device memory for our calculation
  X = new float[len];
  gpu_Y = new float[len * channels];
  nsec = new float[rows];
  dsec = new float[rows];

  cpu_y = new float[len];

  // input
  for (int i = 0; i < len; i++) {
    X[i] = 0.1f;
  }

  // Init value
  for (int i = 0; i < rows; i++) {
    nsec[i] = 0.00002f;
    dsec[i] = 0.00005f;
  }
}

void IirBenchmark::Run() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->ndim = 1;
  lparm->ldims[0] = rows;
  lparm->gdims[0] = channels * rows;

  IIR(len, c, nsec, dsec, sizeof(float) * 512, X, gpu_Y, lparm);
}

void IirBenchmark::Verify() {
  //  Compute CPU results
  float *ds = new float[rows * 2];
  float *ns = new float[rows * 2];

  // internal state
  float *u = new float[rows * 2];
  memset(u, 0, sizeof(float) * rows * 2);

  float out, unew;

  int i, j;

  for (i = 0; i < rows; i++) {
    ds[i * 2] = ds[i * 2 + 1] = 0.00005f;
    ns[i * 2] = ns[i * 2 + 1] = 0.00002f;
  }

  for (i = 0; i < len; i++) {
    out = c * X[i];

    for (j = 0; j < rows; j++) {
      unew = X[i] - (ds[j * 2] * u[j * 2] + ds[j * 2 + 1] * u[j * 2 + 1]);
      u[j * 2 + 1] = u[j * 2];
      u[j * 2] = unew;
      out = out + (u[j * 2] * ns[j * 2] + u[j * 2 + 1] * ns[j * 2 + 1]);
    }

    cpu_y[i] = out;
  }

  // Compare CPU and GPU results
  bool success = true;

  int chn;
  for (chn = 0; chn < channels; chn++) {
    size_t start = chn * len;

    for (i = 0; i < len; i++) {
      if (abs(cpu_y[i] - gpu_Y[i + start]) > 0.001) {
        printf("Failed!\n");
        success = false;
        break;
      }
    }
  }

  if (success)
    printf("Passed!\n");

  delete[] ds;
  delete[] ns;
  delete[] u;
}
