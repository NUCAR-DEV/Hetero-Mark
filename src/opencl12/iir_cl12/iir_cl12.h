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
 * Author: Leiming Yu (yu.lei@husky.neu.edu)
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

#ifndef SRC_OPENCL12_IIR_CL12_IIR_CL12_H_
#define SRC_OPENCL12_IIR_CL12_IIR_CL12_H_

#include "src/common/cl_util/cl_util.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/benchmark/benchmark.h"

#define ROWS 256  // num of parallel subfilters

class ParIIR : public Benchmark {
  // Helper objects
  clHelper::clRuntime *runtime;
  clHelper::clFile *file;

  // svm granuality
  bool svmCoarseGrainAvail;
  bool svmFineGrainAvail;

  // OpenCL resources, auto release
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_command_queue cmdQueue;

  // Parameters
  int len;
  int channels;
  float c;

  float *h_X;
  float *h_Y;

  cl_float2 *nsec;
  cl_float2 *dsec;

  // CPU output for comparison
  float *cpu_y;

  // Memory objects
  // cl_mem d_Mat; // Lenx16x2( 32 intermediate data to merge into 1 final data)
  cl_mem d_X;
  cl_mem d_Y;
  cl_mem d_nsec;
  cl_mem d_dsec;

  // User defined kernels
  cl_kernel kernel_pariir;

  //--- ----------------------------------------------------------------------//
  // Initialize functions
  //--- ----------------------------------------------------------------------//
  void Initialize() override;
  void InitParam();
  void InitCL();
  void InitKernels();
  void InitBuffers();

  //--- ----------------------------------------------------------------------//
  // Clear functions
  //--- ----------------------------------------------------------------------//
  void Cleanup() override;
  void CleanUpBuffers();
  void CleanUpKernels();

  // Run kernels
  void multichannel_pariir();

  // check the results
  void compare();

 public:
  ParIIR();
  ~ParIIR();

  void SetInitialParameters(int l) {
    if (l >= ROWS && ((l % ROWS) == 0)) {
      this->len = l;
    } else {
      std::cout << "Invalid value for signal length = " << l << ".\n";
      std::cout << "The length should be at least " << ROWS;
      std::cout << ", and evenly divisible by " << ROWS << ".\n";
      exit(-1);
    }
  }

  void Run() override;
  void Verify() override {}
  void Summarize() override {}
};

#endif  // SRC_OPENCL12_IIR_CL12_IIR_CL12_H_
