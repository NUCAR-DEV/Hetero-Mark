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

#include "src/common/benchmark/benchmark_runner.h"
#include "src/common/time_measurement/timer_impl.h"
#include "src/common/runtime_helper/runtime_helper.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_runtime_helper.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_error_checker.h"
#include "src/common/command_line_option/command_line_option.h"
#include "src/hsa/memory_copy_hsa/memory_copy_benchmark.h"

int main(int argc, const char **argv) {
  // Setup command line option
  CommandLineOption command_line_option(
      "====== Hetero-Mark Memory Copy Benchmarks (HSA mode) ======",
      "This benchmark copies memory from a buffer to another using the GPU");
  command_line_option.AddArgument("Help", "bool", "false",
      "-h", "--help", "Dump help information");
  command_line_option.AddArgument("Help", "bool", "false",
      "-h", "--help", "Dump help information");

  command_line_option.Parse(argc, argv);
  if (command_line_option.GetArgumentValue("Help")->AsBool()) {
    command_line_option.Help();
    return 0;
  }

  // Runtime helper
  HsaErrorChecker error_checker;
  auto hsa_runtime_helper = std::unique_ptr<HsaRuntimeHelper>(
      new HsaRuntimeHelper(&error_checker));

  // Create and run benchmarks
  TimerImpl timer;
  std::unique_ptr<MemoryCopyBenchmark> benchmark(
    new MemoryCopyBenchmark(hsa_runtime_helper.get()));

  for (int i = 10000; i <= 200000; i = i + 10000) {
    benchmark->SetSize(i);
    benchmark->Initialize();
    double start_time[1000];
    double end_time[1000];
    for (int j = 0; j < 1000; j++) {
      start_time[j] = timer.GetTimeInSec();
      benchmark->Run();
      end_time[j] = timer.GetTimeInSec();
    }

    // Average
    double time_sum = 0.0f;
    for (int j = 0; j < 1000; j++) {
      time_sum += end_time[j] - start_time[j];
    }
    time_sum = time_sum / 1000.0;
    printf("Average time copying %d integers is %.12f\n", i, time_sum);
  }
}
