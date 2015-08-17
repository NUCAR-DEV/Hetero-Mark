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

#include <iostream>
#include "hsa/common/BenchmarkRunner.h"
#include "hsa/common/TimeMeasurement.h"
#include "hsa/common/CommandLineOption.h"
#include "hsa/micro-benchmark/atomicity/AtomicityBenchmark.h"

int main(int argc, const char **argv) {
  // Setup command line option
  CommandLineOption commandLineOption(
      "====== Atomicity Micro-benchmark ======",
      "This benchmark checks if the shared memory writting operation is "
      "atomic. The method of this micro-benchmark is to let multiple work "
      "items to write into the same memory location at the same time. If the "
      "writing time increases linearly with the number of work items, the "
      "writing oepration is serialized.");
  commandLineOption.addArgument("Help",
      "bool", "false",
      "-h", "--help",
      "Dump help information");
  commandLineOption.addArgument("Length", "integer", "10000000",
      "-l", "--length",
      "Length of the memory to write to");
  commandLineOption.addArgument("Max number of work items",
      "integer", "2",
      "-m", "--max-wi",
      "Maximum number of workitems to be tested");

  commandLineOption.parse(argc, argv);
  if (commandLineOption.getArgumentValue("Help")->asBool()) {
    commandLineOption.help();
    return 0;
  }
  int maxNumWorkItem = commandLineOption.getArgumentValue(
      "Max number of work items")->asInt32();
  uint64_t length = commandLineOption.getArgumentValue("Length")->asUInt64();
  std::cout << length << "\n";

  for (int i = 1; i <= maxNumWorkItem; i++) {
    TimeMeasurement timer;
    AtomicityBenchmark benchmark;
    benchmark.setNumWorkItem(i);
    benchmark.setLength(length);
    BenchmarkRunner runner(&benchmark, &timer);
    runner.run();
    double time = timer.getTime("Run");
    std::cout << "Number of work item: " << i << ", time: " << time << "\n";
  }



  return 0;
}
