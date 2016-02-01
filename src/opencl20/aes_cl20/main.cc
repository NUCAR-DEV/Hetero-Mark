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
 * Author: Carter McCardwell (cmccardw@coe.neu.edu)
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

#include <cstdlib>
#include <string>

#include "include/aes_cl20.h"
#include "src/common/benchmark/benchmark_runner.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/time_measurement/time_measurement_impl.h"
#include "src/common/command_line_option/command_line_option.h"

int main(int argc, char const *argv[]) {
  // Setup command line option
  CommandLineOption command_line_option(
      "====== Hetero-Mark AES Benchmarks (OpenCL 2.0) ======",
      "This benchmarks runs the AES Algorithm.");
  command_line_option.AddArgument("Help", "bool", "false", "-h", "--help",
                                  "Dump help information");
  command_line_option.AddArgument("Mode", "string", "h", "-m", "--mode",
                                  "Mode of input parsing, \'h\' interpretes "
                                  "the data in hex format, \'a\' interpretes "
                                  "the data as ASCII");
  command_line_option.AddArgument("InputFile", "string", "in.txt", "-i",
                                  "--input", "The input file to be encrypted");
  command_line_option.AddArgument("Keyfile", "string", "key.txt", "-k",
                                  "--keyfile",
                                  "The file containing the key in hex format");
  command_line_option.AddArgument(
      "OutFile", "string", "out.bin", "-o", "--output",
      "The file where the encrypted output should be written");

  command_line_option.Parse(argc, argv);
  if (command_line_option.GetArgumentValue("Help")->AsBool()) {
    command_line_option.Help();
    return 0;
  }

  FilePackage fp;
  fp.mode = const_cast<char *>(
      command_line_option.GetArgumentValue("Mode")->AsString().c_str());
  fp.in = const_cast<char *>(
      command_line_option.GetArgumentValue("InputFile")->AsString().c_str());
  fp.key = const_cast<char *>(
      command_line_option.GetArgumentValue("Keyfile")->AsString().c_str());
  fp.out = const_cast<char *>(
      command_line_option.GetArgumentValue("OutFile")->AsString().c_str());

  std::unique_ptr<AES> aes(new AES());
  aes->SetInitialParameters(fp);
  aes->Initialize();

  std::unique_ptr<TimeMeasurement> timer(new TimeMeasurementImpl());
  BenchmarkRunner runner(aes.get(), timer.get());
  runner.Run();
  runner.Summarize();

  return 0;
}
