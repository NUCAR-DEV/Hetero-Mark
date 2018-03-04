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

#ifndef SRC_COMMON_BENCHMARK_BENCHMARK_RUNNER_H_
#define SRC_COMMON_BENCHMARK_BENCHMARK_RUNNER_H_

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class BenchmarkRunner {
 public:
  /**
   * Constructor
   */
  BenchmarkRunner(Benchmark *benchmark, TimeMeasurement *time_measurement)
      : benchmark_(benchmark), time_measurement_(time_measurement) {}

  virtual ~BenchmarkRunner() {}

  /**
   * Run the benchmark
   */
  virtual void Run();

  /**
   * Dump summarize
   */
  virtual void Summarize(std::ostream *ostream = &std::cout);

  /**
   * Set if the benchmark should be run in verification mode
   */
  virtual void SetVerificationMode(bool verification_mode) {
    verification_mode_ = verification_mode;
  }

  /**
   * Set to true if you want to runner runs the summary functino
   */
  virtual void SetQuietMode(bool quiet_mode) { quiet_mode_ = quiet_mode; }

  /**
   * Set if you want to dump execution time information
   */
  virtual void SetTimingMode(bool timing_mode) { timing_mode_ = timing_mode; }

  /**
   * Set the number of times that the warmup run will be performed
   */
  virtual void SetWarmUpTime(uint32_t warm_up_time) {
    warm_up_time_ = warm_up_time;
  }

  /**
   * Set the number of times that the benchmark runs
   */
  virtual void SetRepeatTime(uint32_t repeat_time) {
    repeat_time_ = repeat_time;
  }

 protected:
  Benchmark *benchmark_;

  bool verification_mode_ = false;
  bool quiet_mode_ = false;
  bool timing_mode_ = false;
  uint32_t warm_up_time_ = 1;
  uint32_t repeat_time_ = 1;

  TimeMeasurement *time_measurement_;
};

#endif  // SRC_COMMON_BENCHMARK_BENCHMARK_RUNNER_H_
