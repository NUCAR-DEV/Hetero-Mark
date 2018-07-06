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

#include "src/kmeans/kmeans_command_line_options.h"

void KmeansCommandLineOptions::RegisterOptions() {
  BenchmarkCommandLineOptions::RegisterOptions();

  command_line_option_.SetBenchmarkName("KMEANS Benchmark");
  command_line_option_.SetDescription(
      "This benchmarks runs the KMeans Algorithm.");
  command_line_option_.AddArgument("InputFile", "string", "", "-i", "--input",
                                   "File containing data to be clustered");
  command_line_option_.AddArgument("MaxNumClusters", "int", "5", "-m", "--max",
                                   "Maximum number of clusters allowed");
  command_line_option_.AddArgument("MinNumClusters", "int", "5", "-n", "--min",
                                   "Minimum number of clusters allowed");
  command_line_option_.AddArgument("Threshold", "double", "0.001", "",
                                   "--threshold", "Threshold value");
  command_line_option_.AddArgument("NumLoops", "int", "1", "-l", "--loops",
                                   "Iteration for each number of clusters");
  command_line_option_.AddArgument("NumPoints", "int", "0", "-x", "--points",
                                   "Number of points to cluster. Setting this "
                                   "field will disable loading input data from "
                                   "file.");
  command_line_option_.AddArgument(
      "NumFeatures", "int", "0", "-y", "--features",
      "Number of features for a point. It has to be used together with the "
      "NumPoints argument.");
}

void KmeansCommandLineOptions::Parse(int argc, const char *argv[]) {
  BenchmarkCommandLineOptions::Parse(argc, argv);

  filename_ = command_line_option_.GetArgumentValue("InputFile")->AsString();
  threshold_ = command_line_option_.GetArgumentValue("Threshold")->AsDouble();
  max_num_clusters_ =
      command_line_option_.GetArgumentValue("MaxNumClusters")->AsInt32();
  min_num_clusters_ =
      command_line_option_.GetArgumentValue("MinNumClusters")->AsInt32();
  num_loops_ = command_line_option_.GetArgumentValue("NumLoops")->AsInt32();
  num_points_ = command_line_option_.GetArgumentValue("NumPoints")->AsInt32();
  num_features_ = command_line_option_.GetArgumentValue("NumFeatures")->AsInt32();
}

void KmeansCommandLineOptions::ConfigureKmeansBenchmark(
    KmeansBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);
  benchmark->setFilename(filename_);
  benchmark->setThreshold(threshold_);
  benchmark->setMaxNumClusters(max_num_clusters_);
  benchmark->setMinNumClusters(min_num_clusters_);
  benchmark->setNumLoops(num_loops_);
  benchmark->setNumPoints(num_points_);
  benchmark->setNumFeatures(num_features_);
}
