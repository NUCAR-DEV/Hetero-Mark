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

#include "src/fdeb/fdeb_command_line_options.h"

void FdebCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("FDEB Benchmark");
  command_line_option_.SetDescription(
      "This benchmark implements a Force Directed Edge Bundling algorithm.");

  command_line_option_.AddArgument("Input", "string", "1000x2000.csv", "-i",
                                   "--input-file",
                                   "The data file to load from.");
  command_line_option_.AddArgument(
      "Collaborative", "bool", "false", "-c", "--collaborative",
      "When enabled, CPU and GPU execution will overlap.");
  command_line_option_.AddArgument(
      "GPUBatchSize", "int", "4096", "-y", "--batch",
      "The number of points that the GPU processes in one kernel. This "
      "argument is only useful in collaborative mode and atomic is not used.");
  command_line_option_.AddArgument(
      "UseAtomic", "bool", "false", "", "--atomic",
      "If atomic is used, the CPU and GPU will use system level "
      "atomic to communicate. This only works in collaborative "
      "mode.");

  command_line_option_.AddArgument("Cycle", "integer", "6", "", "--cycle",
                                   "Number of cycles to run.");
  command_line_option_.AddArgument(
      "InitIter", "integer", "50", "", "--init-iter",
      "Number of iterations to run in the first cycle. The number of "
      "iterations is reduce by 2/3 for each cycle");
  command_line_option_.AddArgument("KP", "float", "0.2", "", "--kp",
                                   "Kp is a argument that how likely that a "
                                   "subdivision point will be staying on the "
                                   "straight line of the original edge.");
  command_line_option_.AddArgument(
      "InitStepSize", "float", "0.04", "-s", "--step-size",
      "The distance to move for each point. The step size will be reduced "
      "after completing each cycle.");
}

void FdebCommandLineOptions::Parse(int argc, const char *argv[]) {
  try {
    BenchmarkCommandLineOptions::Parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  input_file_ = command_line_option_.GetArgumentValue("Input")->AsString();
  cycle_ = command_line_option_.GetArgumentValue("Cycle")->AsInt32();
  init_iter_ = command_line_option_.GetArgumentValue("InitIter")->AsInt32();
  kp_ = command_line_option_.GetArgumentValue("KP")->AsDouble();
  init_step_size_ =
      command_line_option_.GetArgumentValue("InitStepSize")->AsDouble();
  collaborative_ =
      command_line_option_.GetArgumentValue("Collaborative")->AsBool();
  gpu_batch_ = command_line_option_.GetArgumentValue("GpuBatch")->AsInt32();
  use_atomic_ = command_line_option_.GetArgumentValue("UseAtomic")->AsBool();
}

void FdebCommandLineOptions::ConfigureFdebBenchmark(FdebBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);

  benchmark->SetInputFile(input_file_);
  benchmark->SetCollaborative(collaborative_);

  benchmark->SetNumCycle(cycle_);
  benchmark->SetInitStepSize(init_step_size_);
  benchmark->SetKp(kp_);
  benchmark->SetInitIter(init_iter_);
  benchmark->SetGpuBatch(gpu_batch_);
  benchmark->SetUseAtomic(use_atomic_);
}
