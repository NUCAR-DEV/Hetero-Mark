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

#include "hsa/common/BenchmarkRunner.h"
#include "hsa/common/TimeMeasurement.h"
#include "hsa/common/CommandLineOption.h"
#include "hsa/micro-benchmark/caching/ReadOnCPU.h"
#include "hsa/micro-benchmark/caching/ReadOnGPU.h"
#include "hsa/micro-benchmark/caching/ReadOnCPUTwice.h"
#include "hsa/micro-benchmark/caching/ReadOnGPUTwice.h"

int main(int argc, const char **argv) {
  // Setup command line option
  CommandLineOption commandLineOption("Caching Micro-benchmark",
      "This benchmark checks if HSA uses proper cache for both CPU and GPU"
      " memory access");
  commandLineOption.addArgument("Help", "bool", "false",
      "-h", "--help",
      "Dump help information");
  commandLineOption.addArgument("CPU Length", "integer", "1000000000",
      "", "--cpu-length",
      "Length of the vector to access on CPU, "
      "measured in number of 32-bit integers");
  commandLineOption.addArgument("GPU Length", "integer", "1000000",
      "", "--gpu-length",
      "Length of the vector to access on GPU, "
      "measured in number of 32-bit integers");

  commandLineOption.parse(argc, argv);
  if (commandLineOption.getArgumentValue("Help")->asBool()) {
    commandLineOption.help();
    return 0;
  }
  uint64_t cpuSize = commandLineOption.getArgumentValue("CPU Length")
    ->asUInt64();
  uint64_t gpuSize = commandLineOption.getArgumentValue("GPU Length")
    ->asUInt64();

  // Read on CPU once
  ReadOnCPU readOnCPU;
  readOnCPU.setSize(cpuSize);
  TimeMeasurement timeMeasurement;
  BenchmarkRunner readOnCPURunner(&readOnCPU, &timeMeasurement);
  readOnCPURunner.run();
  double cpu_once = timeMeasurement.getTime("Run");

  // Read on CPU twice
  ReadOnCPUTwice readOnCPUTwice;
  readOnCPUTwice.setSize(cpuSize);
  TimeMeasurement timeMeasurement2;
  BenchmarkRunner readOnCPURunner2(&readOnCPUTwice, &timeMeasurement2);
  readOnCPURunner2.run();
  double cpu_twice = timeMeasurement2.getTime("Run");

  // Read on GPU
  ReadOnGPU readOnGPU;
  readOnGPU.setSize(gpuSize);
  TimeMeasurement timeMeasurement3;
  BenchmarkRunner readOnGPURunner(&readOnGPU, &timeMeasurement3);
  readOnGPURunner.run();
  double gpu_once = timeMeasurement3.getTime("Run");

  // Read on GPU twice
  ReadOnGPUTwice readOnGPUTwice;
  readOnGPUTwice.setSize(gpuSize);
  TimeMeasurement timeMeasurement4;
  BenchmarkRunner readOnGPURunner2(&readOnGPUTwice, &timeMeasurement4);
  readOnGPURunner2.run();
  double gpu_twice = timeMeasurement4.getTime("Run");

  std::cout << "Time to access CPU memory once: " << cpu_once << "\n";
  std::cout << "Time to access CPU memory twice: " << cpu_twice << "\n";
  std::cout << "CPU cache ratio: " << cpu_twice/cpu_once << "\n";
  std::cout << "Time to access GPU memory once: " << gpu_once << "\n";
  std::cout << "Time to access GPU memory twice: " << gpu_twice << "\n";
  std::cout << "GPU cache ratio: " << gpu_twice/gpu_once << "\n";

  return 0;
}
