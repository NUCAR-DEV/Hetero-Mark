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
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/time_measurement/time_measurement_impl.h"
#include "src/common/command_line_option/command_line_option.h"
#include "src/hsa/kmeans_hsa/kmeans_benchmark.h"

int main(int argc, const char **argv) {
  // Setup command line option
  CommandLineOption command_line_option(
    "====== Hetero-Mark KMEANS Benchmark "
    "(HSA mode) ======",
    "This benchmarks runs kmeans algorithm.");
  command_line_option.AddArgument("Help", "bool", "false",
      "-h", "--help", "Dump help information");
  command_line_option.AddArgument("Input File", "string", "input.txt",
      "-i", "--input-file",
      "The file that containing data to be clustered");
  command_line_option.AddArgument("Maximum Clusters", "integer",
      "5",
      "-m", "--max-clusters",
      "Maximum number of clusters allowed");
  command_line_option.AddArgument("Minimum Clusters", "integer",
      "5",
      "-n", "--min-clusters",
      "Minimum number of clusters allowed");
  command_line_option.AddArgument("Threshold", "float",
      "0.001",
      "-t", "--threshold",
      "Threshold value");
  command_line_option.AddArgument("Number Loops", "integer",
      "1",
      "-l", "--loops",
      "Number of loops");
  command_line_option.AddArgument("Verify", "bool", "false",
      "-v", "--verify",
      "Verify the calculation result");

  command_line_option.Parse(argc, argv);
  if (command_line_option.GetArgumentValue("Help")->AsBool()) {
    command_line_option.Help();
    return 0;
  }

  bool verify = command_line_option.GetArgumentValue("Verify")->AsBool();
  std::string input = command_line_option.GetArgumentValue("Input File")
    ->AsString();
  uint32_t n_loops = command_line_option.GetArgumentValue("Number Loops")
    ->AsUInt32();
  double threshold = command_line_option.GetArgumentValue("Threshold")
    ->AsDouble();
  uint32_t max_clusters = command_line_option.GetArgumentValue(
    "Maximum Clusters")->AsUInt32();
  uint32_t min_clusters = command_line_option.GetArgumentValue(
    "Minimum Clusters")->AsUInt32();

  // Create and setup benchmarks
  std::unique_ptr<KmeansBenchmark> benchmark(new KmeansBenchmark());
  benchmark->SetInputFileName(input.c_str());
  benchmark->SetNLoops(n_loops);
  benchmark->SetThreshold(threshold);
  benchmark->SetMaxClusters(max_clusters);
  benchmark->SetMinClusters(min_clusters);

  // Run benchmark
  std::unique_ptr<TimeMeasurement> timer(new TimeMeasurementImpl());
  BenchmarkRunner runner(benchmark.get(), timer.get());
  runner.set_verification_mode(verify);
  runner.Run();
  runner.Summarize();
}
