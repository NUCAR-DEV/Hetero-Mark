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

#include "src/pr/pr_command_line_options.h"

void PrCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("PR Benchmark");
  command_line_option_.SetDescription("This benchmark runs PR.");

  command_line_option_.AddArgument("MaxIteration", "integer", "500", "-m",
                                   "--max-iteration",
                                   "Maximum number of iterations to converge");

  command_line_option_.AddArgument("InputFile", "string", "", "-i",
                                   "--input-file",
                                   "The input file that contains the network");
}

void PrCommandLineOptions::Parse(int argc, const char *argv[]) {
  BenchmarkCommandLineOptions::Parse(argc, argv);

  max_iteration_ =
      command_line_option_.GetArgumentValue("MaxIteration")->AsUInt32();

  input_file_name_ =
      command_line_option_.GetArgumentValue("InputFile")->AsString();
}

void PrCommandLineOptions::ConfigurePrBenchmark(PrBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);
  benchmark->SetMaxIteration(max_iteration_);
  benchmark->SetInputFileName(input_file_name_.c_str());
}
