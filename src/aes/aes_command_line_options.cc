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

#include "src/aes/aes_command_line_options.h"

void AesCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("AES Benchmark");
  command_line_option_.SetDescription(
      "This benchmark runs AES encryption algorithm");

  command_line_option_.AddArgument("InputFile", "string", "", "-i",
                                   "--input-file",
                                   "Path to input file that contains the "
                                   "plaintext to be encrypted");

  command_line_option_.AddArgument("KeyFile", "string", "", "-k", "--key-file",
                                   "Path to the file that contains the "
                                   "key to be used to encrypt plaintext");

  command_line_option_.AddArgument(
      "InputLength", "integer", "0", "-x", "--length",
      "The input plain text length in bytes. Setting this field will disable "
      "loading the key and the input from files.");
}

void AesCommandLineOptions::Parse(int argc, const char *argv[]) {
  BenchmarkCommandLineOptions::Parse(argc, argv);

  input_file_ = command_line_option_.GetArgumentValue("InputFile")->AsString();
  key_file_ = command_line_option_.GetArgumentValue("KeyFile")->AsString();
  input_length_ = command_line_option_.GetArgumentValue("InputLength")->AsUInt64();
}

void AesCommandLineOptions::ConfigureAesBenchmark(AesBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);

  benchmark->SetInputFileName(input_file_);
  benchmark->SetKeyFileName(key_file_);
  benchmark->SetInputLength(input_length_);
}
