/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 *   DEALINGS WITH THE SOFTWARE.
 *
 * Hidden Markov Model with OpenCL 2.0
 *
 * It takes number of hidden states as argument
 *
 */

#include "hmm_cl20.h"
#include <stdint.h>/* for uint64 definition */
#include <time.h>/* for clock_gettime */
#include <string.h>
#include <math.h>
#include <iostream>
#include <clUtil.h>

using namespace std;

#define BILLION 1000000000L

HMM::HMM(int N) {
  if (N >= TILE) {
    this->N = N;
  } else {
    std::cout << "N < " << TILE << std::endl;
    exit(-1);
  }
}

HMM::~HMM() {
   // Cleanup auto-called
}

void HMM::Init() {
  InitParam();
  InitCL();
  InitKernels();
  InitBuffers();
}

void HMM::InitCL() {
  // Init OCL context
  runtime    = clRuntime::getInstance();

  // OpenCL objects get from clRuntime class release themselves automatically,
  // no need to clRelease them explicitly
  platform   = runtime->getPlatformID();
  device     = runtime->getDevice();
  context    = runtime->getContext();

  cmdQueue_0 = runtime->getCmdQueue(0);
  // cmdQueue_1 = runtime->getCmdQueue(1);

  // Helper to read kernel file
  file       = clFile::getInstance();
}

void HMM::InitParam() {
  if (N) {
    bytes_nn       = sizeof(float) * N * N;
    bytes_nt       = sizeof(float) * N * T;
    bytes_n        = sizeof(float) * N;
    bytes_dt       = sizeof(float) * D * T;
    bytes_dd       = sizeof(float) * D * D;
    bytes_dn       = sizeof(float) * D * N;
    bytes_ddn      = sizeof(float) * D * D * N;
    bytes_t        = sizeof(float) * T;
    bytes_d        = sizeof(float) * D;
    bytes_n        = sizeof(float) * N;
    bytes_const    = sizeof(float) * 4096;  // 16 KB
    dd             = D * D;

    tileblks       = (N/TILE) * (N/TILE);  // [N/16][N/16]
    bytes_tileblks = sizeof(float) * tileblks;

    blk_rows       = D/16;
    blknum         = blk_rows * (blk_rows + 1) / 2;
  } else {
    std::cout << "Invalid N" << std::endl;
    exit(-1);
  }
}

void HMM::InitKernels() {
  cl_int err;

  file->open("hmm_Kernels.cl");

  // Create program
  const char *source = file->getSourceChar();

  program = clCreateProgramWithSource(context, 1,
                                      (const char **)&source, NULL, &err);
  checkOpenCLErrors(err, "Failed to create Program with source...\n");

  // Create program with OpenCL 2.0 support
  err = clBuildProgram(program, 0, NULL, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");

  // Program build info
  // char buf[0x10000];
  // clGetProgramBuildInfo( program,
  //                         device,
  //                         CL_PROGRAM_BUILD_LOG,
  //                         0x10000,
  //                         buf,
  //                         NULL);
  // printf("\n%s\n", buf);

  // Create kernels
  // Forward
  kernel_FWD_init_alpha = clCreateKernel(program, "FWD_init_alpha", &err);
  checkOpenCLErrors(err, "Failed to create kernel FWD_init_alpha")

    kernel_FWD_norm_alpha = clCreateKernel(program, "FWD_norm_alpha", &err);
  checkOpenCLErrors(err, "Failed to create kernel FWD_norm_alpha")

    kernel_TransposeSym = clCreateKernel(program, "TransposeSym", &err);
  checkOpenCLErrors(err, "Failed to create kernel TransposeSym")

    kernel_FWD_update_alpha = clCreateKernel(program, "FWD_update_alpha", &err);
  checkOpenCLErrors(err, "Failed to create kernel FWD_update_alpha")

    // Backward
    kernel_BK_BetaB = clCreateKernel(program, "BK_BetaB", &err);
  checkOpenCLErrors(err, "Failed to create kernel BK_BetaB")

    kernel_BK_update_beta = clCreateKernel(program, "BK_update_beta", &err);
  checkOpenCLErrors(err, "Failed to create kernel BK_update_beta")

    kernel_BK_norm_beta = clCreateKernel(program, "BK_norm_beta", &err);
  checkOpenCLErrors(err, "Failed to create kernel BK_norm_beta")

    // EM
    kernel_EM_betaB_alphabeta = clCreateKernel(program,
                                               "EM_betaB_alphabeta", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_betaB_alphabeta")

    kernel_EM_update_gamma = clCreateKernel(program, "EM_update_gamma", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_update_gamma")

    kernel_EM_alpha_betaB = clCreateKernel(program, "EM_alpha_betaB", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_alpha_betaB")

    kernel_EM_pre_xisum = clCreateKernel(program, "EM_pre_xisum", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_pre_xisum")

    kernel_EM_update_xisum = clCreateKernel(program, "EM_update_xisum", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_update_xisum")

    kernel_EM_gamma = clCreateKernel(program, "EM_gamma", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_gamma")

    kernel_EM_expectA = clCreateKernel(program, "EM_expectA", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_expectA")

    kernel_EM_gamma_state_sum = clCreateKernel(program,
                                               "EM_gamma_state_sum", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_gamma_state_sum")

    kernel_EM_gamma_obs = clCreateKernel(program, "EM_gamma_obs", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_gamma_obs")

    kernel_EM_expect_mu = clCreateKernel(program, "EM_expect_mu", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_expect_mu")

    kernel_EM_sigma_dev = clCreateKernel(program, "EM_sigma_dev", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_sigma_dev")

    kernel_EM_expect_sigma = clCreateKernel(program, "EM_expect_sigma", &err);
  checkOpenCLErrors(err, "Failed to create kernel EM_expect_sigma")
}

void HMM::InitBuffers() {
  // CPU buffers
  // SVM buffers
  //        a,b,prior,lll, blk_result

  cl_int err;
  int i, j;

  bool svmCoarseGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_COARSE);
  bool svmFineGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_FINE);

  // Need at least coarse grain
  if (!svmCoarseGrainAvail) {
    printf("SVM coarse grain support unavailable\n");
    exit(-1);
  }

  // Alloc buffer
  // if (!svmFineGrainAvail)
  // {
  //        printf("SVM fine grain support unavailable\n");

  // Prepare
  // state transition probability matrix
  a = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

  // emission probability matrix
  b = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);

  // forward probability matrix: TxN
  alpha = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);

  // prior probability
  prior = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);

  // observed input
  observations = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dt, 0);

  // Constant memory
  constMem = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_const, 0);

  // Forward
  // log likelihood
  lll = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float), 0);
  aT  = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

  // Backward
  beta  = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
  betaB = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);

  // EM
  xi_sum     = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
  alpha_beta = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
  gamma      = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
  alpha_betaB = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
  xi_sum_tmp  = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

  // intermediate blk results from the device
  blk_result = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                   bytes_tileblks, 0);

  // Expected values
  expect_prior = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
  expect_A     = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
  expect_mu    = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dn, 0);
  expect_sigma = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_ddn, 0);

  gamma_state_sum = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
  gamma_obs       = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                        bytes_dt, 0);

  // }
  // else
  // {
  //        printf("Not implemented\n");
  //        exit(-1);
  // }

  // Sanity check
  if (!a || !b || !alpha || !prior || !blk_result || !observations || !lll) {
    printf("Cannot allocate SVM memory with clSVMAlloc\n");
    exit(-1);
  }

  // Inistilize Input Data
  // Coarse grain SVM needs explicit map/unmap
  // if (!svmFineGrainAvail)
  // {
  // Map a
  err = clEnqueueSVMMap(cmdQueue_0,
                        CL_TRUE,       // blocking map
                        CL_MAP_WRITE,
                        a,
                        bytes_nn,
                        0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

  // Map b
  err = clEnqueueSVMMap(cmdQueue_0,
                        CL_TRUE,       // blocking map
                        CL_MAP_WRITE,
                        b,
                        bytes_nt,
                        0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

  // Map prior
  err = clEnqueueSVMMap(cmdQueue_0,
                        CL_TRUE,       // blocking map
                        CL_MAP_WRITE,
                        prior,
                        bytes_n,
                        0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

  // Map observations
  err = clEnqueueSVMMap(cmdQueue_0,
                        CL_TRUE,       // blocking map
                        CL_MAP_WRITE,
                        observations,
                        bytes_dt,
                        0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");
  //}

  // Init content
  for (i = 0; i < (N * N); i++)
    a[i] = 1.0f/(float)N;

  for (i = 0; i < (N * T); i++)
    b[i] = 1.0f/(float)T;

  for (i = 0; i < N; i++)
    prior[i] = 1.0f/(float)N;

  // D x T
  for (i = 0 ; i < D; ++i)
    for (j = 0 ; j< T; ++j)
      observations[i * T + j] = (float)j + 1.f;


  // Coarse grain needs explicit unmap
  // if (!svmFineGrainAvail)
  // {
  err = clEnqueueSVMUnmap(cmdQueue_0, a, 0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");

  err = clEnqueueSVMUnmap(cmdQueue_0, b, 0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");

  err = clEnqueueSVMUnmap(cmdQueue_0, prior, 0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");

  err = clEnqueueSVMUnmap(cmdQueue_0, observations, 0, 0, 0);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
  //}
}

void HMM::Cleanup() {
  CleanUpKernels();
  CleanUpBuffers();
}

#define safeSVMFree(ctx, ptr) \
  if (ptr)                     \
    clSVMFree(ctx, ptr);
void HMM::CleanUpBuffers() {
  // CPU buffers
  safeSVMFree(context, a);
  safeSVMFree(context, b);
  safeSVMFree(context, alpha);
  safeSVMFree(context, prior);
  safeSVMFree(context, observations);

  // Forward
  safeSVMFree(context, lll);
  safeSVMFree(context, aT);

  // Backward
  safeSVMFree(context, beta);
  safeSVMFree(context, betaB);

  // EM
  safeSVMFree(context, xi_sum);
  safeSVMFree(context, alpha_beta);
  safeSVMFree(context, gamma);
  safeSVMFree(context, alpha_betaB);
  safeSVMFree(context, xi_sum_tmp);
  safeSVMFree(context, blk_result);

  safeSVMFree(context, expect_prior);
  safeSVMFree(context, expect_A);
  safeSVMFree(context, expect_mu);
  safeSVMFree(context, expect_sigma);

  safeSVMFree(context, gamma_state_sum);
  safeSVMFree(context, gamma_obs);
}
#undef safeSVMFree

void HMM::CleanUpKernels() {
  // Forward
  checkOpenCLErrors(clReleaseKernel(kernel_FWD_init_alpha),
                    "Failed to release kernel kernel_FWD_init_alpha");

  checkOpenCLErrors(clReleaseKernel(kernel_FWD_norm_alpha),
                    "Failed to release kernel kernel_FWD_norm_alpha");

  checkOpenCLErrors(clReleaseKernel(kernel_TransposeSym),
                    "Failed to release kernel kernel_TransposeSym");

  checkOpenCLErrors(clReleaseKernel(kernel_FWD_update_alpha),
                    "Failed to release kernel kernel_FWD_update_alpha");

  // Backward
  checkOpenCLErrors(clReleaseKernel(kernel_BK_BetaB),
                    "Failed to release kernel kernel_BK_BetaB");

  checkOpenCLErrors(clReleaseKernel(kernel_BK_update_beta),
                    "Failed to release kernel kernel_BK_update_beta");

  checkOpenCLErrors(clReleaseKernel(kernel_BK_norm_beta),
                    "Failed to release kernel kernel_BK_norm_beta");

  // EM
  checkOpenCLErrors(clReleaseKernel(kernel_EM_betaB_alphabeta),
                    "Failed to release kernel kernel_EM_betaB_alphabeta");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_update_gamma),
                    "Failed to release kernel kernel_EM_update_gamma");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_alpha_betaB),
                    "Failed to release kernel kernel_EM_alpha_betaB");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_pre_xisum),
                    "Failed to release kernel kernel_EM_pre_xisum");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_update_xisum),
                    "Failed to release kernel kernel_EM_update_xisum");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_gamma),
                    "Failed to release kernel kernel_EM_gamma");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_expectA),
                    "Failed to release kernel kernel_EM_expectA");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_gamma_state_sum),
                    "Failed to release kernel kernel_EM_gamma_state_sum");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_gamma_obs),
                    "Failed to release kernel kernel_EM_gamma_obs");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_expect_mu),
                    "Failed to release kernel kernel_EM_expect_mu");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_sigma_dev),
                    "Failed to release kernel kernel_EM_sigma_dev");

  checkOpenCLErrors(clReleaseKernel(kernel_EM_expect_sigma),
                    "Failed to release kernel kernel_EM_expect_sigma");
}

//                                    Run Forward()
void HMM::Forward() {
  // clear lll
  float zero = 0.f;
  clEnqueueSVMMemFill(cmdQueue_0,
                      lll,
                      (void *)&zero,
                      sizeof(float),
                      sizeof(float),
                      0, NULL, NULL);

  ForwardInitAlpha();

  ForwardNormAlpha(0);

  TransposeSym(a, aT, N);

  int frm;
  int current, previous;

  for (frm = 1; frm < T; ++frm) {
    current  = frm * N;
    previous = current - N;

    // b. * (a' * alpha)
    // copy alpha to constant memory first
    clEnqueueSVMMemcpy(cmdQueue_0,
                       CL_TRUE,
                       constMem,
                       &alpha[previous],
                       bytes_n,
                       0, NULL, NULL);

    ForwardUpdateAlpha(current);

    // Normalize alpha at current frame
    // Update log likelihood
    ForwardNormAlpha(current);
  }
}

//                                    Run Backward()
void HMM::Backward() {
  int j;
  int current, previous;
  // TODO: xi_sum and gamma update could be run concurrently
  for (j = T-2; j >= 0; --j) {
    current = j * N;
    previous = current + N;

    // beta(t+1) .* b(t+1)
    BackwardBetaB(previous);

    // copy betaB to constant memory
    clEnqueueSVMMemcpy(cmdQueue_0,
                       CL_TRUE,
                       constMem,
                       betaB,
                       bytes_n,
                       0, NULL, NULL);

    // beta(t-1) = a * betaB
    BackwardUpdateBeta(current);
    // normalize beta at current frame
    BackwardNormBeta(current);
  }
}

//                                    Run EM()
void HMM::EM() {
  cl_int err;

  // clear data for xi_sum
  float zero = 0.f;
  clEnqueueSVMMemFill(cmdQueue_0,
                      xi_sum,
                      (void *)&zero,
                      sizeof(float),
                      bytes_nn,
                      0, NULL, NULL);

  float sum;

  int i, current, previous;
  int window;

  for (window = 0; window < (T - 1); ++window) {
      current = window * N;
      previous = current + N;

      // compute beta(t+1) * B(t+1) and alpha(t) * beta(t)
      EM_betaB_alphabeta(current, previous);

      // normalise alpha_beta, upate gamma for current frame
      EM_update_gamma(current);

      // alpha * betaB'
      EM_alpha_betaB(current);

      // normalise( A .*  (alpha * betaB') )
      // compute xi_sum_tmp and block results
      // FIXME: sometimes segmentation faults
      EM_pre_xisum();

      sum = 0.f;
      // Map to host to finalized the summation
      if (!svmFineGrainAvail) {
        err = clEnqueueSVMMap(cmdQueue_0, CL_TRUE, CL_MAP_READ,
                              blk_result, bytes_tileblks, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");
      }

      // compute on cpu
#pragma unroll
      for (i = 0; i < tileblks; ++i) {
        sum += blk_result[i];
      }

      if (!svmFineGrainAvail) {
        err = clEnqueueSVMUnmap(cmdQueue_0, blk_result, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
      }

      // cout << sum << endl;

      // Update the xi_sum
      EM_update_xisum(sum);
  }

  // update gamma at the last frame
  current = previous;

  // TODO: reuse template
  EM_gamma(current);

  // update expected values
  // TODO: Merge with previous gamma update
  // expected_prior = gamma(:, 1);

  // check from here !!!
  err = clEnqueueSVMMemcpy(cmdQueue_0,
                           CL_TRUE,
                           expect_prior,
                           gamma,
                           bytes_n, 0, NULL, NULL);

  checkOpenCLErrors(err, "Failed to copy gamma to expect_prior");

  // update expect_A
  EM_expectA();

  // compute gamma_state_sum
  // gamma is T x N, gamma_state_sum is the colum-wise summation
  EM_gamma_state_sum();

  // transpose observations
  // TODO: Concurrent Kernel Execution
  size_t start;
  int hs;
  for (hs = 0 ; hs < N; ++hs) {
    // copy gamma(hs,:) to constant memory
    clEnqueueSVMMemcpy(cmdQueue_0,
                       CL_TRUE,
                       constMem,
                       &gamma[hs * T],
                       bytes_t,
                       0, NULL, NULL);

    // compute gammaobs
    EM_gamma_obs();

    current = hs * D;

    // compute expect_mu
    // TODO: map to host when there is not enough data?
    EM_expect_mu(current, hs);

    // copy expect_mu to constant mem
    clEnqueueSVMMemcpy(cmdQueue_0,
                       CL_TRUE,
                       constMem,
                       &expect_mu[hs * D],
                       bytes_d,
                       0, NULL, NULL);

    // compute sigma_dev
    EM_sigma_dev(hs);

    start =  hs * dd;

    // update expect_sigma
    EM_expect_sigma(start);
  }
}

//                                        Forward Functions

void HMM::ForwardInitAlpha() {
  cl_int err;

  size_t globalSize = (size_t)(ceil(N/256.f) * 256);
  size_t localSize = 256;

  err = clSetKernelArg(kernel_FWD_init_alpha, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 1, (void *)b);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 2, (void *)prior);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 3, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 4, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");


  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_FWD_init_alpha,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::ForwardNormAlpha(int startpos) {
  cl_int err;

  size_t localSize = 256;
  size_t globalSize = (size_t)(ceil(N/256.f) * 256);

  int pos = startpos;

  err = clSetKernelArg(kernel_FWD_norm_alpha, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_norm_alpha, 1, sizeof(int), (void *)&pos);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_norm_alpha, 2,
                       sizeof(float)*256, NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_FWD_norm_alpha, 3, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_norm_alpha, 4, (void *)lll);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_FWD_norm_alpha,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::TransposeSym(float *a, float *aT, int size) {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_TransposeSym, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_TransposeSym, 1,
                       sizeof(float)*272, NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_TransposeSym, 2, (void *)a);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_TransposeSym, 3, (void *)aT);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_TransposeSym,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::ForwardUpdateAlpha(int pos) {
  int current = pos;

  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {16, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_FWD_update_alpha, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 1,
                       sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 2,
                       sizeof(float)*272,  NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_FWD_update_alpha, 3, (void *)constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_update_alpha, 4, (void *)aT);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_update_alpha, 5, (void *)b);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_FWD_update_alpha, 6, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_FWD_update_alpha,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

//                                  Backward Functions
void HMM::BackwardBetaB(int pos) {
  int previous = pos;

  cl_int err;

  size_t globalSize = (size_t)(ceil(N/256.f) * 256);
  size_t localSize = 256;

  err = clSetKernelArg(kernel_BK_BetaB, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_BetaB, 1, sizeof(int), (void *)&previous);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_BK_BetaB, 2, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_BK_BetaB, 3, (void *)b);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_BK_BetaB, 4, (void *)betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_BK_BetaB,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);

  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}


void HMM::BackwardUpdateBeta(int pos) {
  int current = pos;

  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {16, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_BK_update_beta, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_update_beta, 1, sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_update_beta, 2,
                       sizeof(float)*272,  NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 3, (void *)constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 4, (void *)a);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 5, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_BK_update_beta,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::BackwardNormBeta(int pos) {
  cl_int err;

  size_t localSize = 256;
  size_t globalSize = (size_t)(ceil(N/256.f) * 256);

  int current = pos;

  err = clSetKernelArg(kernel_BK_norm_beta, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_norm_beta, 1, sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_norm_beta, 2,
                       sizeof(float)*256, NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_BK_norm_beta, 3, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_BK_norm_beta,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

//                                        EM Functions
void HMM::EM_betaB_alphabeta(int curpos, int prepos) {
  cl_int err;

  int current = curpos;
  int previous = prepos;

  size_t localSize = 256;
  size_t globalSize = (size_t)(ceil(N/256.f) * 256);

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 1,
                       sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 2,
                       sizeof(int), (void *)&previous);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 3, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 4, (void *)b);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 5, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 6, (void *)betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 7,
                                (void *)alpha_beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");


  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_betaB_alphabeta,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_update_gamma(int pos) {
  cl_int err;

  int current = pos;

  size_t localSize = 256;
  size_t globalSize = (size_t)(ceil(N/256.f) * 256);


  err = clSetKernelArg(kernel_EM_update_gamma, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_update_gamma, 1,
                       sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_update_gamma, 2, sizeof(float) * 256, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_update_gamma, 3, (void *)alpha_beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_update_gamma, 4, (void *)gamma);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_update_gamma,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_alpha_betaB(int pos) {
  cl_int err;

  int current = pos;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_EM_alpha_betaB, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_alpha_betaB, 1, sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_alpha_betaB, 2, (void *)betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_alpha_betaB, 3, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_alpha_betaB, 4, (void *)alpha_betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_alpha_betaB,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_pre_xisum() {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_EM_pre_xisum, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_pre_xisum, 1, sizeof(float) * 272, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_pre_xisum, 2, (void *)a);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_pre_xisum, 3, (void *)alpha_betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_pre_xisum, 4, (void *)xi_sum_tmp);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_pre_xisum, 5, (void *)blk_result);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_pre_xisum,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_update_xisum(float sumvalue) {
  float sum = sumvalue;

  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_EM_update_xisum, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_update_xisum, 1, sizeof(float), (void *)&sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_update_xisum, 2, (void *)xi_sum_tmp);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_update_xisum, 3, (void *)xi_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_update_xisum,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_gamma(int pos) {
  cl_int err;

  int current = pos;

  size_t localSize = 256;
  size_t globalSize = (size_t)(ceil(N/256.f) * 256);

  err = clSetKernelArg(kernel_EM_gamma, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_gamma, 1, sizeof(int), (void *)&current);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_gamma, 2, sizeof(float) * 256, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma, 3, (void *)alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma, 4, (void *)beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma, 5, (void *)gamma);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_gamma,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_expectA() {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {16, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_EM_expectA, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expectA, 1, sizeof(float) * 272, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_expectA, 2, (void *)xi_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_expectA, 3, (void *)expect_A);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_expectA,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_gamma_state_sum() {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, 16};  // N is multiple of 16

  err = clSetKernelArg(kernel_EM_gamma_state_sum, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_gamma_state_sum, 1, sizeof(int), (void *)&T);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_gamma_state_sum, 2, sizeof(float) * 272, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma_state_sum, 3, (void *)gamma);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma_state_sum, 4,
                                 (void *)gamma_state_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_gamma_state_sum,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_gamma_obs() {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)T, (size_t)D};  // T and D is multiple of 16

  err = clSetKernelArg(kernel_EM_gamma_obs, 0, sizeof(int), (void *)&D);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_gamma_obs, 1, sizeof(int), (void *)&T);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma_obs, 2, (void *)constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma_obs, 3, (void *)observations);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_gamma_obs, 4, (void *)gamma_obs);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_gamma_obs,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_expect_mu(int pos, int currentstate) {
  int offset = pos;
  int hs = currentstate;

  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {16, (size_t)D};  // D is multiple of 16

  err = clSetKernelArg(kernel_EM_expect_mu, 0, sizeof(int), (void *)&D);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_mu, 1, sizeof(int), (void *)&T);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_mu, 2, sizeof(int), (void *)&offset);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_mu, 3, sizeof(int), (void *)&hs);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_mu, 4, sizeof(float) * 272, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_expect_mu, 5, (void *)gamma_obs);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_expect_mu, 6,
                                 (void *)gamma_state_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_expect_mu, 7, (void *)expect_mu);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_expect_mu,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_sigma_dev(int currentstate) {
  int hs = currentstate;

  cl_int err;

  size_t localSize[2]  = {8, 8};
  size_t globalSize[2] = {(size_t)D, (size_t)D};  // D is multiple of 16

  err = clSetKernelArg(kernel_EM_sigma_dev, 0, sizeof(int), (void *)&D);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_sigma_dev, 1, sizeof(int), (void *)&T);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_sigma_dev, 2, sizeof(int), (void *)&hs);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_sigma_dev, 3, (void *)constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_sigma_dev, 4, (void *)gamma_obs);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_sigma_dev, 5, (void *)observations);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_sigma_dev, 6,
                                 (void *)gamma_state_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_sigma_dev, 7, (void *)sigma_dev);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_sigma_dev,
                               2,
                               0, globalSize, localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EM_expect_sigma(size_t pos) {
  cl_int err;

  size_t start = pos;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {16, (size_t)blknum};

  // TODO
  err = clSetKernelArg(kernel_EM_expect_sigma, 0,
                       sizeof(int), (void *)&blk_rows);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_sigma, 1,
                       sizeof(int), (void *)&D);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_sigma, 2,
                       sizeof(size_t), (void *)&start);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArgSVMPointer(kernel_EM_expect_sigma, 3, (void *)sigma_dev);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArgSVMPointer(kernel_EM_expect_sigma, 4,
                                 (void *)expect_sigma);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_EM_expect_sigma,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0,
                               NULL,
                               NULL);
  checkOpenCLErrors(err, "Failed to execute kernel!");
}

//                                      Run HMM
void HMM::Run() {
  // HMM Parameters
  //        a,b,prior,alpha
  printf("=>Initialize parameters.\n");
  Init();

  // Forward Algorithm on GPU
  printf("\n");
  printf("      >> Start  Forward Algorithm on GPU.\n");
  Forward();
  printf("      >> Finish Forward Algorithm on GPU.\n");

  // Backward Algorithm on GPU
  printf("\n");
  printf("      >> Start  Backward Algorithm on GPU.\n");
  Backward();
  printf("      >> Finish Backward Algorithm on GPU.\n");

  // EM Algorithm on GPU
  printf("\n");
  printf("      >> Start  EM Algorithm on GPU.\n");
  EM();
  printf("      >> Finish EM Algorithm on GPU.\n");

  printf("<=End program.\n");
}