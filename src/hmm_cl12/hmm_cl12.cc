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
 * Hidden Markov Model with OpenCL 1.2
 *
 * It takes number of hidden states as argument
 *
 */

#include "hmm_cl12.h"

#include <stdint.h>/* for uint64 definition */
#include <time.h>/* for clock_gettime */
#include <math.h>
#include <string.h>
#include <iostream>
#include "src/common/cl_util/cl_util.h"

using namespace std;

#define BILLION 1000000000L

HMM::HMM() {
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
  file->open("hmm_cl12_Kernels.cl");

  // Create program
  const char *source = file->getSourceChar();

  program = clCreateProgramWithSource(context, 1,
                                      (const char **)&source, NULL, &err);
  checkOpenCLErrors(err, "Failed to create Program with source...\n");

  // Create program with OpenCL 1.2 support
  err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0x10000,
                          buf,
                          NULL);
    printf("\n%s\n", buf);
    exit(-1);
  }

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

  // Prepare
  // state transition probability matrix
  a = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: a ");

  // emission probability matrix
  b = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: b ");

  // forward probability matrix: TxN
  alpha = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: alpha ");

  // prior probability
  prior = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_n, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: prior ");

  // observed input
  observations = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                bytes_dt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: observations ");

  // Constant memory
  constMem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                            bytes_const, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: constMem ");

  // Forward
  // log likelihood
  lll = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: lll ");
  aT  = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: aT ");

  // Backward
  beta  = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: beta ");
  betaB = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_n, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: betaB ");

  // EM
  xi_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: xi_sum ");
  alpha_beta = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_n, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: alpha_beta ");
  gamma = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: gamma ");
  alpha_betaB = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: alpha_betaB ");
  xi_sum_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: xi_sum_tmp ");

  // intermediate blk results from the device
  blk_result = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              bytes_tileblks, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: blk_result ");

  // Expected values
  expect_prior = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                bytes_n, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: expect_prior ");
  expect_A = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_nn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: expect_A ");
  expect_mu = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_dn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: expect_mu ");
  expect_sigma = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                bytes_ddn, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: expect_sigma ");

  gamma_state_sum = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   bytes_n, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: gamma_state_sum ");
  gamma_obs = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   bytes_dt, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: gamma_obs ");

  // Intitialize Input Data, explicit copy
  // C++11 smart pointers
  std::unique_ptr<float[]> a_host(new float[N*N]);
  std::unique_ptr<float[]> b_host(new float[N*T]);
  std::unique_ptr<float[]> prior_host(new float[N]);
  std::unique_ptr<float[]> observations_host(new float[D*T]);

  for (i = 0; i < (N * N); i++)
    a_host.get()[i] = 1.0f/(float)N;
  for (i = 0; i < (N * T); i++)
    b_host.get()[i] = 1.0f/(float)T;
  for (i = 0; i < N; i++)
    prior_host.get()[i] = 1.0f/(float)N;
  for (i = 0 ; i < D; ++i)
    for (j = 0 ; j< T; ++j)
      observations_host.get()[i * T + j] = (float)j + 1.f;  // D x T

  // OCL 1.2 needs explicit copy
  err = clEnqueueWriteBuffer(cmdQueue_0,
                             a,
                             CL_TRUE,
                             0,
                             sizeof(float)*N*N,
                             (void *)a_host.get(),
                             0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to copy data to a");

  err = clEnqueueWriteBuffer(cmdQueue_0,
                             b,
                             CL_TRUE,
                             0,
                             sizeof(float)*N*T,
                             (void *)b_host.get(),
                             0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to copy data to b");

  err = clEnqueueWriteBuffer(cmdQueue_0,
                             prior,
                             CL_TRUE,
                             0,
                             sizeof(float)*N,
                             (void *)prior_host.get(),
                             0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to copy data to prior");

  err = clEnqueueWriteBuffer(cmdQueue_0,
                             observations,
                             CL_TRUE,
                             0,
                             sizeof(float)*D*T,
                             (void *)observations_host.get(),
                             0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to copy data to observations");

  clFinish(cmdQueue_0);
}

void HMM::Cleanup() {
  CleanUpKernels();
  CleanUpBuffers();
}

#define clFreeBuffer(buf) \
  err = clReleaseMemObject(buf); \
  checkOpenCLErrors(err, "Failed to release cl buffer");
void HMM::CleanUpBuffers() {
  cl_int err;

  // CPU buffers
  clFreeBuffer(a);
  clFreeBuffer(b);
  clFreeBuffer(alpha);
  clFreeBuffer(prior);
  clFreeBuffer(observations);

  // Forward
  clFreeBuffer(lll);
  clFreeBuffer(aT);

  // Backward
  clFreeBuffer(beta);
  clFreeBuffer(betaB);

  // EM
  clFreeBuffer(xi_sum);
  clFreeBuffer(alpha_beta);
  clFreeBuffer(gamma);
  clFreeBuffer(alpha_betaB);
  clFreeBuffer(xi_sum_tmp);
  clFreeBuffer(blk_result);

  clFreeBuffer(expect_prior);
  clFreeBuffer(expect_A);
  clFreeBuffer(expect_mu);
  clFreeBuffer(expect_sigma);

  clFreeBuffer(gamma_state_sum);
  clFreeBuffer(gamma_obs);
}
#undef clFreeBuffer

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
  clEnqueueFillBuffer(cmdQueue_0,
                      lll,
                      (void *)&zero,
                      sizeof(float),
                      0,
                      sizeof(float),
                      0, NULL, NULL);

  ForwardInitAlpha();

  ForwardNormAlpha(0);

  TransposeSym(N);

  int frm;
  int current, previous;

  for (frm = 1; frm < T; ++frm) {
    current  = frm * N;
    previous = current - N;

    // b. * (a' * alpha)
    // copy alpha to constant memory first
    clEnqueueCopyBuffer(cmdQueue_0,
                        alpha,                   // src buffer
                        constMem,                // dst buffer
                        sizeof(float)*previous,  // src offset
                        0,                       // dst offset
                        bytes_n,                 // size to copy
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
      clEnqueueCopyBuffer(cmdQueue_0,
                          betaB,     // src buffer
                          constMem,  // dst buffer
                          0,         // src offset
                          0,         // dst offset
                          bytes_n,   // bytes to copy
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
  clEnqueueFillBuffer(cmdQueue_0,
                      xi_sum,
                      (void *)&zero,
                      sizeof(float),
                      0,
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

    // Explicitly copy to host to finalized the summation
    std::unique_ptr<float[]> blk_result_host(new float[tileblks]);

    err = clEnqueueReadBuffer(cmdQueue_0,
                              blk_result,
                              CL_TRUE,
                              0,
                              bytes_tileblks,
                              (void *)blk_result_host.get(),
                              0, NULL, NULL);
    checkOpenCLErrors(err, "Failed to clEnqueueCopyBuffer");

    // compute on cpu
#pragma unroll
      for (i=0; i< tileblks; ++i)
        sum += blk_result_host.get()[i];

      // cout << sum << endl;

      // Update the xi_sum
      EM_update_xisum(sum);
    }

  // update gamma at the last frame
  current = previous;

  // TODO(): reuse template
  EM_gamma(current);

  // update expected values

  // TODO: Merge with previous gamma update
  // expected_prior = gamma(:, 1);
  // check from here !!!
  err = clEnqueueCopyBuffer(cmdQueue_0,
                            gamma,         // src buffer
                            expect_prior,  // dst buffer
                            0,             // src offset
                            0,             // dst offset
                            bytes_n,       // bytes to copy
                            0, NULL, NULL);

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
    clEnqueueCopyBuffer(cmdQueue_0,
                        gamma,               // src buffer
                        constMem,            // dst buffer
                        sizeof(float)*hs*T,  // src offset
                        0,                   // dst offset
                        bytes_t,             // size to copy
                        0, NULL, NULL);

      // compute gammaobs
    EM_gamma_obs();
    current = hs * D;

    // compute expect_mu
    // TODO: map to host when there is not enough data?
    EM_expect_mu(current, hs);

    // copy expect_mu to constant mem
    clEnqueueCopyBuffer(cmdQueue_0,
                        expect_mu,           // src buffer
                        constMem,            // dst buffer
                        sizeof(float)*hs*D,  // src offset
                        0,                   // dst offset
                        bytes_d,             // size to copy
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

  err = clSetKernelArg(kernel_FWD_init_alpha, 1, sizeof(cl_mem), (void *)&b);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clSetKernelArg(kernel_FWD_init_alpha, 2,
                       sizeof(cl_mem), (void *)&prior);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clSetKernelArg(kernel_FWD_init_alpha, 3,
                       sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clSetKernelArg(kernel_FWD_init_alpha, 4, sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

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

  err = clSetKernelArg(kernel_FWD_norm_alpha, 2, sizeof(float)*256, NULL);
  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_norm_alpha, 3,
                       sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clSetKernelArg(kernel_FWD_norm_alpha, 4, sizeof(cl_mem), (void *)&lll);
  checkOpenCLErrors(err, "Failed at Kernel Arguments.");

  err = clEnqueueNDRangeKernel(cmdQueue_0,
                               kernel_FWD_norm_alpha,
                               1,
                               0, &globalSize, &localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::TransposeSym(int size) {
  cl_int err;

  size_t localSize[2]  = {16, 16};
  size_t globalSize[2] = {(size_t)N, (size_t)N};  // N is multiple of 16

  err = clSetKernelArg(kernel_TransposeSym, 0, sizeof(int), (void *)&N);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_TransposeSym, 1, sizeof(float)*272, NULL);
  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_TransposeSym, 2, sizeof(cl_mem), (void *)&a);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_TransposeSym, 3, sizeof(cl_mem), (void *)&aT);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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
                       sizeof(float)*272, NULL);  // local memory
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 3,
                       sizeof(cl_mem), (void *)&constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 4,
                       sizeof(cl_mem), (void *)&aT);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 5,
                       sizeof(cl_mem), (void *)&b);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_FWD_update_alpha, 6,
                       sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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

  err = clSetKernelArg(kernel_BK_BetaB, 2, sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_BetaB, 3, sizeof(cl_mem), (void *)&b);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_BetaB, 4, sizeof(cl_mem), (void *)&betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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

  err = clSetKernelArg(kernel_BK_update_beta, 3,
                       sizeof(cl_mem), (void *)&constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_update_beta, 4, sizeof(cl_mem), (void *)&a);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_BK_update_beta, 5, sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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

  err = clSetKernelArg(kernel_BK_norm_beta, 3, sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 3,
                       sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 4,
                       sizeof(cl_mem), (void *)&b);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 5,
                       sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 6,
                       sizeof(cl_mem), (void *)&betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_betaB_alphabeta, 7,
                       sizeof(cl_mem), (void *)&alpha_beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

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

  err = clSetKernelArg(kernel_EM_update_gamma, 2,
                       sizeof(float) * 256, NULL);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_update_gamma, 3,
                       sizeof(cl_mem), (void *)&alpha_beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_update_gamma, 4,
                       sizeof(cl_mem), (void *)&gamma);
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

  err = clSetKernelArg(kernel_EM_alpha_betaB, 2,
                       sizeof(cl_mem), (void *)&betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_alpha_betaB, 3,
                       sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_alpha_betaB, 4,
                       sizeof(cl_mem), (void *)&alpha_betaB);
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

  err = clSetKernelArg(kernel_EM_pre_xisum, 2, sizeof(cl_mem), (void *)&a);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_pre_xisum, 3,
                       sizeof(cl_mem), (void *)&alpha_betaB);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_pre_xisum, 4,
                       sizeof(cl_mem), (void *)&xi_sum_tmp);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_pre_xisum, 5,
                       sizeof(cl_mem), (void *)&blk_result);
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

  err = clSetKernelArg(kernel_EM_update_xisum, 2,
                       sizeof(cl_mem), (void *)&xi_sum_tmp);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_update_xisum, 3,
                       sizeof(cl_mem), (void *)&xi_sum);
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

  err = clSetKernelArg(kernel_EM_gamma, 3, sizeof(cl_mem), (void *)&alpha);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_gamma, 4, sizeof(cl_mem), (void *)&beta);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_gamma, 5, sizeof(cl_mem), (void *)&gamma);
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

  err = clSetKernelArg(kernel_EM_expectA, 2, sizeof(cl_mem), (void *)&xi_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_expectA, 3, sizeof(cl_mem), (void *)&expect_A);
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

  err = clSetKernelArg(kernel_EM_gamma_state_sum, 3,
                       sizeof(cl_mem), (void *)&gamma);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_gamma_state_sum, 4,
                       sizeof(cl_mem), (void *)&gamma_state_sum);
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

  err = clSetKernelArg(kernel_EM_gamma_obs, 2,
                       sizeof(cl_mem), (void *)&constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_gamma_obs, 3,
                       sizeof(cl_mem), (void *)&observations);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_gamma_obs, 4,
                       sizeof(cl_mem), (void *)&gamma_obs);
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

  err = clSetKernelArg(kernel_EM_expect_mu, 5,
                       sizeof(cl_mem), (void *)&gamma_obs);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_expect_mu, 6,
                       sizeof(cl_mem), (void *)&gamma_state_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_expect_mu,
                       7, sizeof(cl_mem), (void *)&expect_mu);
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

  err = clSetKernelArg(kernel_EM_sigma_dev, 3,
                       sizeof(cl_mem), (void *)&constMem);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_sigma_dev, 4,
                       sizeof(cl_mem), (void *)&gamma_obs);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_sigma_dev, 5,
                       sizeof(cl_mem), (void *)&observations);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_sigma_dev, 6,
                       sizeof(cl_mem), (void *)&gamma_state_sum);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_sigma_dev, 7,
                       sizeof(cl_mem), (void *)&sigma_dev);
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

  // FIXME: no cast?
  int start = (int)pos;

  // FIXME: OCL 1.2 doesn't support non-uniform workgroup size,
  // i.e global size must be multiples of local size
  // size_t localSize[2]  = {16, 16};
  size_t localSize[2]  = {16, (size_t)blknum};
  size_t globalSize[2] = {16, (size_t)blknum};

  // TODO
  err = clSetKernelArg(kernel_EM_expect_sigma, 0,
                       sizeof(int), (void *)&blk_rows);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_sigma, 1, sizeof(int), (void *)&D);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_sigma, 2, sizeof(int), (void *)&start);
  checkOpenCLErrors(err, "Failed at clSetKernelArg");

  err = clSetKernelArg(kernel_EM_expect_sigma, 3,
                       sizeof(cl_mem), (void *)&sigma_dev);
  checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

  err = clSetKernelArg(kernel_EM_expect_sigma, 4,
                       sizeof(cl_mem), (void *)&expect_sigma);
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
