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

#include "src/hsa/search_and_use/search_and_use_benchmark.h"

#include <iostream>

SearchAndUseBenchmark::SearchAndUseBenchmark(
  HsaRuntimeHelper *runtime_helper) :
  runtime_helper_(runtime_helper){
}

void SearchAndUseBenchmark::Initialize() {
  // Initialize runtime
  runtime_helper_->InitializeOrDie();

  // Find GPU device
  agent_ = runtime_helper_->FindGpuOrDie();
  std::cout << agent_->GetNameOrDie() << "\n";

  // Create program from source
  executable_ = runtime_helper_->CreateProgramFromSourceOrDie(
      "kernels.brig", agent_);

  // Get Kernel
  kernel_ = executable_->GetKernel("&__OpenCL_vector_copy_kernel", 
      agent_);
  
  // Create a queue
  queue_ = agent_->CreateQueueOrDie();

  // Prepare buffers 
  in_ = new float[1024];
  for (unsigned int i = 0; i < 1024; i++) {
    in_[i] = (float)i;
  }
  out_ = new float[1024];

  // Prepare to launch kernel
  kernel_->SetDimension(1);
  kernel_->SetLocalSize(1, 256);
  kernel_->SetGlobalSize(1, 1024);
  kernel_->SetKernelArgument(1, sizeof(in_), &in_);
  kernel_->SetKernelArgument(2, sizeof(out_), &out_);
}

void SearchAndUseBenchmark::Run() {
  kernel_->ExecuteKernel(agent_, queue_);
}

void SearchAndUseBenchmark::Verify() {
}

void SearchAndUseBenchmark::Summarize() {
  for (int i = 0; i < 1024; i++) {
    std::cout << "Index: " << i << ", value: " << out_[i] << ".\n";
  }
}

void SearchAndUseBenchmark::Cleanup() {
  delete[] in_;
  delete[] out_;
}

