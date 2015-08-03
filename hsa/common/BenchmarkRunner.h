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

#ifndef HSA_COMMON_BENCHMARKRUNNER_H_
#define HSA_COMMON_BENCHMARKRUNNER_H_

#include "hsa/common/Benchmark.h"
#include "hsa/common/TimeMeasurement.h"

class BenchmarkRunner {
 public:
  /**
   * Constructor
   */
  BenchmarkRunner(Benchmark *benchmark, TimeMeasurement *timeMeasurement) :
    benchmark(benchmark),
    timeMeasurement(timeMeasurement) {
  }

  /**
   * Run the benchmark
   */
  virtual void run();

  /**
   * Dump summarize
   */
  virtual void summarize(std::ostream *ostream = &std::cout);

  /**
   * Set if the benchmark should be run in verification mode
   */
  virtual void setVerificationMode(bool verificationMode) {
    this->verificationMode = verificationMode;
  }

 protected:
  // The benchmark to run
  Benchmark *benchmark;

  // Is verification mode
  bool verificationMode = false;

  // Time measurement
  TimeMeasurement *timeMeasurement;
};

#endif  // HSA_COMMON_BENCHMARKRUNNER_H_