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

#include "src/be/be_command_line_options.h"

void BeCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("Background Extraction Benchmark");
  command_line_option_.SetDescription(
      "This benchmark runs Background "
      "Extraction Algorithm. It learns the background of the video, and "
      "substract the background from each frame");

  command_line_option_.AddArgument("InputFile", "string", "", "-i",
                                   "--input-video",
                                   "The video to be processed by the "
                                   "background extractor.");
  command_line_option_.AddArgument("MaxFrames", "integer", "0", "-m",
                                   "--max-frames",
                                   "Max number of frames to process. 0 means "
                                   "no limitation.");
  command_line_option_.AddArgument("Collaborative", "bool", "false", "-c",
                                   "--collaborative",
                                   "When enabled, the CPU will fetch and "
                                   "decode while GPU is doing back ground "
                                   "extraction.");
  command_line_option_.AddArgument("GenerateOutput", "bool", "false", "-g",
                                   "--generate-output",
                                   "When enabled, the CPU execution result "
                                   "and the GPU result will be stored in "
                                   "output videos.");
}

void BeCommandLineOptions::Parse(int argc, const char *argv[]) {
  BenchmarkCommandLineOptions::Parse(argc, argv);

  input_file_ = command_line_option_.GetArgumentValue("InputFile")->AsString();
  max_frame_ = command_line_option_.GetArgumentValue("MaxFrames")->AsUInt32();
  collaborative_execution_ =
      command_line_option_.GetArgumentValue("Collaborative")->AsBool();
  generate_output_ =
      command_line_option_.GetArgumentValue("GenerateOutput")->AsBool();
}

void BeCommandLineOptions::ConfigureBeBenchmark(BeBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);

  benchmark->SetInputFile(input_file_);
  benchmark->SetMaxFrame(max_frame_);
  benchmark->SetCollaborativeExecution(collaborative_execution_);
  benchmark->SetGenerateOutput(generate_output_);
}
