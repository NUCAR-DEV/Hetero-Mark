/*
 * Copyright (c) 2015 Northeastern University
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
 */

#ifndef SRC_FIR_CL20_FIR_CL20_BENCHMARK_H_
#define SRC_FIR_CL20_FIR_CL20_BENCHMARK_H_

#include "src/common/cl_util/cl_benchmark.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/fir/fir_benchmark.h"

class FirCl20Benchmark : public FirBenchmark, public ClBenchmark {
 private:
  cl_kernel fir_kernel_;

  cl_float *input_svm_ = NULL;
  cl_float *output_svm_ = NULL;
  cl_float *history_ = NULL;
  cl_float *coeff_svm_ = NULL;

  void InitializeData();
  void InitializeKernels();
  void InitializeBuffers();

  void MapSvmBuffers();
  void UnmapSvmBuffers();

 public:
  FirCl20Benchmark(){}
  ~FirCl20Benchmark(){}

  void Initialize() override;
  void Run() override;
  void Cleanup() override;
  void Summarize() override {}
};

#endif  // SRC_FIR_CL20_FIR_CL20_BENCHMARK_H_
