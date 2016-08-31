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

#include "src/ga/ga_command_line_options.h"

void GaCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("GA Benchmark");
  command_line_option_.SetDescription("This benchmark runs GA.");

  command_line_option_.AddArgument("InputFile", "string", "", "-i",
                                   "--input-file", "The input file.");
  command_line_option_.AddArgument("Collaborative", "bool", "false",
                                   "", "--collaborative", 
                                   "Enable CPU GPU collaborative execution.");
}

void GaCommandLineOptions::Parse(int argc, const char *argv[]) {
  BenchmarkCommandLineOptions::Parse(argc, argv);

  input_file_ = command_line_option_.GetArgumentValue("InputFile")->AsString();
  collaborative_ = command_line_option_.GetArgumentValue("Collaborative")->AsBool();
}

void GaCommandLineOptions::ConfigureBenchmark(GaBenchmark *benchmark) {
  benchmark->SetInputFile(input_file_);
  benchmark->SetCollaborativeExecution(collaborative_);
}
