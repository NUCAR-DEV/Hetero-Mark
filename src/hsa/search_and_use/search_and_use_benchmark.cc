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
#include <thread>

SearchAndUseBenchmark::SearchAndUseBenchmark(
    HsaRuntimeHelper *runtime_helper, 
    Timer *timer) :
  runtime_helper_(runtime_helper),
  timer_(timer) {
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
  signal_ = runtime_helper_->CreateSignal(0);

  // Prepare to launch kernel
  kernel_->SetDimension(1);
  kernel_->SetLocalSize(1, 256);
  kernel_->SetGlobalSize(1, 256 * 20);
  kernel_->SetKernelArgument(1, sizeof(in_), &in_);
  kernel_->SetKernelArgument(2, sizeof(out_), &out_);
  kernel_->SetKernelArgument(3, 8, signal_->GetNative());
}

void SearchAndUseBenchmark::Run() {
  std::thread signal_waiting_thread = 
    std::thread( [this] { WaitForSignal(); } );
  kernel_->ExecuteKernel(agent_, queue_);
  std::cout << "Kernel stopped: " << timer_->GetTimeInSec() << "\n";
  kernel_stopped_ = true;
  signal_waiting_thread.join();
}

void SearchAndUseBenchmark::WaitForSignal() {
  int64_t signal_value = 0;
  std::cout << "Waiting thread started: " << timer_->GetTimeInSec() << "\n";
  while (!kernel_stopped_) {
    signal_value = signal_->WaitForCondition("NE", signal_value);
    std::cout << "Return: " << timer_->GetTimeInSec() << 
      ", value: " << signal_value <<"\n";
  }
}

void SearchAndUseBenchmark::Verify() {
}

void SearchAndUseBenchmark::Summarize() {
}

void SearchAndUseBenchmark::Cleanup() {
  delete[] in_;
  delete[] out_;
}

