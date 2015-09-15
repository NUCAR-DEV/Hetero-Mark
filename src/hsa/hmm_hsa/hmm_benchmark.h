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

#ifndef SRC_HSA_HMM_HSA_HMM_BENCHMARK_H_
#define SRC_HSA_HMM_HSA_HMM_BENCHMARK_H_

#include <cstring>
#include "src/common/benchmark/benchmark.h"

#define __constant const

class HmmBenchmark : public Benchmark {
  // Parameters
  static const int TILE = 16;
  static const int SIZE = 4096;
  static const int BLOCKSIZE = 256;

  int N;
  int T;  // number of (overlapping) windows
  int D;  // number of features

  int bytes_nn;
  int bytes_nt;
  int bytes_dt;
  int bytes_dd;
  int bytes_dn;
  int bytes_ddn;
  int bytes_t;
  int bytes_d;
  int bytes_n;
  int bytes_const;
  int dd;

  int tileblks;
  size_t bytes_tileblks;

  int blk_rows;
  int blknum;

  // OCL 1.2 buffers
  // Prepare
  float *a;             // state transition probability matrix
  float *b;             // emission probability matrix
  float *alpha;         // forward probability matrix
  float *prior;         // prior probability
  float *observations;  // D x T

  // Forward
  float *lll;  // log likelihood
  float *aT;   // transpose of a

  // Backward
  float *beta;
  float *betaB;

  // EM
  float *xi_sum;       // N x N
  float *alpha_beta;   // N
  float *gamma;        // T x N
  float *alpha_betaB;  // N x N
  float *xi_sum_tmp;   // N x N
  float *blk_result;   // intermediate blk results

  float *expect_prior;  // N
  float *expect_A;      // N xN
  float *expect_mu;     // N x D
  float *expect_sigma;  // N x D x D

  float *gamma_state_sum;  // N
  float *gamma_obs;        // D x T
  float *sigma_dev;        // D x D

  // Constant
  float *constMem;

  // Initialize functions
  void InitParam();
  void InitBuffers();

  // Clean functions
  void CleanUp();
  void CleanUpBuffers();

  // Forward functions
  void Forward();
  void ForwardInitAlpha();
  void ForwardNormAlpha(int startpos);
  void TransposeSym(int size);
  void ForwardUpdateAlpha(int pos);

  // Backward functions
  void Backward();
  void BackwardBetaB(int pos);
  void BackwardUpdateBeta(int pos);
  void BackwardNormBeta(int pos);

  // EM functions
  void EM();
  void EM_betaB_alphabeta(int curpos, int prepos);
  void EM_update_gamma(int pos);
  void EM_alpha_betaB(int pos);
  void EM_pre_xisum();
  void EM_update_xisum(float sumvalue);
  void EM_gamma(int pos);
  void EM_expectA();
  void EM_gamma_state_sum();
  void EM_gamma_obs();
  void EM_expect_mu(int pos, int currentstate);
  void EM_sigma_dev(int currentstate);
  void EM_expect_sigma(size_t pos);

 public:
  explicit HmmBenchmark(int N);

  void Initialize() override;
  void Run() override;
  void Verify() override {}
  void Summarize() override {}
  void Cleanup() override;
};

#endif  // SRC_HSA_HMM_HSA_HMM_BENCHMARK_H_
