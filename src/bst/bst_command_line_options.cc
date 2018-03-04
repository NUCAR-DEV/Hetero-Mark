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

#include "src/bst/bst_command_line_options.h"

void BstCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("BST Benchmark");
  command_line_option_.SetDescription(
      "This benchmark runs the Binary Insertion Tree Algorithm.");

  command_line_option_.AddArgument("NumNodes", "integer", "200", "-n",
                                   "--num-nodes",
                                   "Number of nodes to be inserted");

  command_line_option_.AddArgument("InitialPosition", "integer", "10", "-i",
                                   "--init-num-nodes",
                                   "Number of initial nodes");

  command_line_option_.AddArgument(
      "HostPercentage", "integer", "30", "-p", "--host-percentage",
      "Percentage of nodes to be processed by the host");
}

void BstCommandLineOptions::Parse(int argc, const char *argv[]) {
  try {
    BenchmarkCommandLineOptions::Parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  num_insert_ = command_line_option_.GetArgumentValue("NumNodes")->AsUInt32();

  init_tree_insert_ =
      command_line_option_.GetArgumentValue("InitialPosition")->AsUInt32();

  host_percentage_ =
      command_line_option_.GetArgumentValue("HostPercentage")->AsUInt32();
}

void BstCommandLineOptions::ConfigureBstBenchmark(BstBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);
  benchmark->SetNumNodes(num_insert_);
  benchmark->SetInitPosition(init_tree_insert_);
  benchmark->SetHostPercentage(host_percentage_);
}
