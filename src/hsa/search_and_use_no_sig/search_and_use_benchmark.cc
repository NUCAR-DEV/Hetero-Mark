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
  signal_ = runtime_helper_->CreateSignal(0);

  // Prepare to launch kernel
  kernel_->SetDimension(1);
  kernel_->SetLocalSize(1, 256);
  kernel_->SetGlobalSize(1, 256 * 150);
  kernel_->SetKernelArgument(1, 8, signal_->GetNative());
}

void SearchAndUseBenchmark::Run() {
  kernel_->QueueKernel(agent_, queue_);
  printf("Kernel started: %f\n", timer_->GetTimeInSec());
  kernel_->WaitKernel();
  printf("Kernel stopped: %f\n", timer_->GetTimeInSec());
  kernel_stopped_ = true;
  std::thread signal_waiting_thread = 
  std::thread( [this] { WaitForSignal(); } );
  signal_waiting_thread.join();
}

void SearchAndUseBenchmark::WaitForSignal() {
  std::vector<std::unique_ptr<std::thread>> threads;
  printf("Waiting thread started: %f\n", timer_->GetTimeInSec());
  for(int i = 0; i < 150; i++) {
    auto signal_processing_thread = std::unique_ptr<std::thread> (
      new std::thread( [this] {ProcessSignal();} ));
    threads.push_back(std::move(signal_processing_thread));
  }

  // Wait all thread finish
  for(auto &thread : threads) {
    thread->join();
  }
}

void SearchAndUseBenchmark::ProcessSignal() {
  double start = timer_->GetTimeInSec();
  for (uint64_t i = 0; i < 50000000; i++) {}
  double end = timer_->GetTimeInSec();
  printf("(%f, %f),\n", start, end); 
}

void SearchAndUseBenchmark::Verify() {
}

void SearchAndUseBenchmark::Summarize() {
}

void SearchAndUseBenchmark::Cleanup() {
}

