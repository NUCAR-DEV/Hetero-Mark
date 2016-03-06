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
 * Author: Xiangyu Li (xili@coe.neu.edu)
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

#ifndef SRC_OPENCL12_PAGERANK_CL12_PAGERANK_CL12_H_
#define SRC_OPENCL12_PAGERANK_CL12_PAGERANK_CL12_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include "src/common/cl_util/cl_util.h"
#include "src/common/benchmark/benchmark.h"

class PageRank : public Benchmark {
 private:
  clHelper::clRuntime* runtime;
  clHelper::clFile* file;

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue cmdQueue;

  cl_program program;
  cl_kernel kernel;
  cl_int err;

  void InitKernel();
  void InitBuffer();
  void InitBufferCpu();
  void InitBufferGpu();
  void FillBuffer();
  void FillBufferCpu();
  void FillBufferGpu();
  void ExecKernel();

  void FreeKernel();
  void FreeBuffer();
  void ReadBuffer();

  void ReadCsrMatrix();
  void ReadDenseVector();
  void PageRankCpu();

  void InitCl();

  std::string fileName1;
  std::string fileName2;
  int nnz;
  int nr;
  int* rowOffset;
  int* col;
  float* val;
  float* vector;
  float* eigenV;
  cl_mem d_rowOffset;
  cl_mem d_col;
  cl_mem d_val;
  cl_mem d_vector;
  cl_mem d_eigenV;
  std::ifstream csrMatrix;
  std::ifstream denseVector;
  size_t global_work_size[1];
  size_t local_work_size[1];
  int workGroupSize;
  int maxIter;
  int isVectorGiven;

 public:
  PageRank();

  ~PageRank();
  void SetInitialParameters(std::string fName1, std::string fName2);
  void SetInitialParameters(std::string fName1);
  void Initialize() override;
  void Run() override;
  void Verify() override {}
  void Cleanup() override {}
  void Summarize() override {}

  void CpuRun();
  void Test();
  float* GetEigenV();
  void Print();
  void PrintOutput();
  int GetLength();
  float abs(float value);
};

#endif  // SRC_OPENCL12_PAGERANK_CL12_PAGERANK_CL12_H_
