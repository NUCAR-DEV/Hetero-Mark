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

#include "src/common/command_line_option/benchmark_command_line_options.h"
#include <cstdlib>

void BenchmarkCommandLineOptions::RegisterOptions() {
  command_line_option_.AddArgument("Help", "bool", "false", "-h", "--help",
                                   "Dump help information");

  command_line_option_.AddArgument("Quiet", "bool", "false", "-q", "--quiet",
                                   "Mute output");

  command_line_option_.AddArgument("Verify", "bool", "false", "-v", "--verify",
                                   "Verify the GPU result with CPU");

  command_line_option_.AddArgument("Timing", "bool", "false", "-t", "--timing",
                                   "Show timing information");

  command_line_option_.AddArgument("Repeat", "integer", "1", "-r", "--repeat",
                                   "Repeat the benchmark execution.");

  command_line_option_.AddArgument(
      "WarmUp", "integer", "1", "-w", "--warm-up",
      "Run the benchmarks for a certain times before measuring time.");

  command_line_option_.AddArgument("MemType", "string", "array", "-m", "--mem",
                                   "The memory manager to use.");
}

void BenchmarkCommandLineOptions::Parse(int argc, const char *argv[]) {
  command_line_option_.Parse(argc, argv);

  DumpHelpOnRequest();

  quiet_mode_ = command_line_option_.GetArgumentValue("Quiet")->AsBool();
  verification_ = command_line_option_.GetArgumentValue("Verify")->AsBool();
  timing_ = command_line_option_.GetArgumentValue("Timing")->AsBool();
  repeat_times_ = command_line_option_.GetArgumentValue("Repeat")->AsUInt32();
  warm_up_times_ = command_line_option_.GetArgumentValue("WarmUp")->AsUInt32();
  mem_type_ = command_line_option_.GetArgumentValue("MemType")->AsString();
}

void BenchmarkCommandLineOptions::DumpHelpOnRequest() {
  if (command_line_option_.GetArgumentValue("Help")->AsBool()) {
    command_line_option_.Help();
    exit(0);
  }
}

void BenchmarkCommandLineOptions::ConfigureBenchmarkRunner(
    BenchmarkRunner *benchmark_runner) {
  benchmark_runner->SetVerificationMode(verification_);
  benchmark_runner->SetQuietMode(quiet_mode_);
  benchmark_runner->SetTimingMode(timing_);
  benchmark_runner->SetRepeatTime(repeat_times_);
  benchmark_runner->SetWarmUpTime(warm_up_times_);
}

void BenchmarkCommandLineOptions::ConfigureBenchmark(Benchmark *benchmark) {
  benchmark->SetMemType(mem_type_);
}
