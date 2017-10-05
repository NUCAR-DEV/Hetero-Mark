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

#include "src/knn/knn_command_line_options.h"

void KnnCommandLineOptions::RegisterOptions() {
  // Register General Options
  BenchmarkCommandLineOptions::RegisterOptions();

  // The name of the Benchmark -- BlackScholes
  command_line_option_.SetBenchmarkName("KNN Benchmark");

  // The description of the benchmark
  command_line_option_.SetDescription(
      "This benchmark runs k nearest neighbours application on input -lat latitiude_value -lng longitude_value");

  // Registering the command line options

  command_line_option_.AddArgument("InputFile", "string","", "-i", "--input",
                                   "The filename that lists the data input files");
  command_line_option_.AddArgument("Latitude", "float", "0", "-lat","-latitude",
                                   "The latitude for the nearest neighbours");
  command_line_option_.AddArgument("Longitude", "float", "0", "-lng","-longitude",
                                   "The longitude for the nearest neighbours");
  command_line_option_.AddArgument("kValue", "int", "10","-k", "-numK",
		  		    "The number of nearest neighbours you want");
}

void KnnCommandLineOptions::Parse(int argc, const char *argv[]) {
  // Parse general Options
  BenchmarkCommandLineOptions::Parse(argc, argv);

  // Parse the input arguments for number of elements for KNN
  filename_ =
      command_line_option_.GetArgumentValue("InputFile")->AsString();

  latitude_ =
      command_line_option_.GetArgumentValue("Latitude")->AsDouble();

  longitude_ =
      command_line_option_.GetArgumentValue("Longitude")->AsDouble();

  k_value_=
      command_line_option_.GetArgumentValue("kValue")->AsInt32();
}

void KnnCommandLineOptions::ConfigureKnnBenchmark(KnnBenchmark *benchmark) {
  BenchmarkCommandLineOptions::ConfigureBenchmark(benchmark);

  // Call the setter to set the number of elements
  benchmark->setFilename(filename_);
  benchmark->setLatitude(latitude_);
  benchmark->setLongitude(longitude_);
  benchmark->setKValue(k_value_);
}
