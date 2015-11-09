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


#include "include/kmeans_cl20.h"
#include "src/common/benchmark/benchmark_runner.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/time_measurement/time_measurement_impl.h"
#include "src/common/command_line_option/command_line_option.h"

#include <cstdlib>
#include <string>

int main(int argc, char const *argv[])
{
    // Setup command line option
    CommandLineOption command_line_option(
      "====== Hetero-Mark KMeans Benchmarks (OpenCL 2.0) ======",
      "This benchmarks runs the KMeans Algorithm.");
    command_line_option.AddArgument("Help", "bool", "false",
        "-h", "--help", "Dump help information");
    command_line_option.AddArgument("FileName", "string", "",
        "-i", "--input",
        "File containing data to be clustered");
    command_line_option.AddArgument("max_nclusters", "int", "5",
        "-m", "--max",
        "Maximum number of clusters allowed");
    command_line_option.AddArgument("min_nclusters", "int", "5",
        "-n", "--min",
        "Minimum number of clusters allowed");
    command_line_option.AddArgument("Threshold", "double", "0.001",
        "-t", "--threshold",
        "Threshold value");
    command_line_option.AddArgument("nloops", "int", "1",
        "-l", "--loops",
        "Iteration for each number of clusters");
    command_line_option.AddArgument("binary", "bool", "false",
        "-b", "--binary",
        "Input file is in binary format");
    command_line_option.AddArgument("rmse", "bool", "false",
        "-r", "--rmse",
        "Calculate RMSE");
    command_line_option.AddArgument("cluster", "bool", "false",
        "-o", "--outputcluster",
        "Output cluster center coordinates");

    FilePackage fp;
    try {
      command_line_option.Parse(argc, argv);

      // Print help information
      if (command_line_option.GetArgumentValue("Help")->AsBool()) {
        command_line_option.Help();
        return 0;
      }

      fp.filename = const_cast<char*>(command_line_option.GetArgumentValue("FileName")->AsString().c_str());
      fp.binary = command_line_option.GetArgumentValue("binary")->AsBool();
      fp.threshold = command_line_option.GetArgumentValue("Threshold")->AsDouble();
      fp.max_cl = command_line_option.GetArgumentValue("max_nclusters")->AsInt32();
      fp.min_cl = command_line_option.GetArgumentValue("min_nclusters")->AsInt32();
      fp.RMSE = command_line_option.GetArgumentValue("rmse")->AsBool();
      fp.output = command_line_option.GetArgumentValue("cluster")->AsBool();
      fp.nloops = command_line_option.GetArgumentValue("nloops")->AsInt32();
    } catch (const std::exception &e) {
      std::cerr << e.what();
      return 0;
    }

    std::unique_ptr<KMEANS> km(new KMEANS());
    km->SetInitialParameters(fp);

    std::unique_ptr<TimeMeasurement> timer(new TimeMeasurementImpl());
    BenchmarkRunner runner(km.get(), timer.get());
    runner.Run();
    runner.Summarize();

    return 0;
}
