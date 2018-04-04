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

#ifndef SRC_COMMON_BENCHMARK_BENCHMARK_H_
#define SRC_COMMON_BENCHMARK_BENCHMARK_H_

#include <memory>
#include <string>

#include "src/common/time_measurement/cpu_gpu_activity_logger.h"
#include "src/common/time_measurement/time_measurement.h"

/**
 * A benchmark is a program that test platform performance. It follows the
 * steps of Initialize, Run, Verify, Summarize and Cleanup.
 */
class Benchmark {
 protected:
  TimeMeasurement *timer_;

  std::unique_ptr<CPUGPUActivityLogger> cpu_gpu_logger_;

  bool quiet_mode_ = false;

  // WorkGroup Size
  uint32_t work_group_size = 256;

  // Number of compute units
  uint32_t num_compute_units_ = 8;

  // The memory type to use
  std::string mem_type_;

 public:
  Benchmark() { cpu_gpu_logger_.reset(new CPUGPUActivityLogger()); }

  virtual ~Benchmark() {}

  /**
   * Initialize environment, parameter, buffers
   */
  virtual void Initialize() = 0;

  /**
   * Run the benchmark
   */
  virtual void Run() = 0;

  /**
   * Verify
   */
  virtual void Verify() = 0;

  /**
   * Summarize
   */
  virtual void Summarize() = 0;

  /**
   * Clean up
   */
  virtual void Cleanup() = 0;

  /**
   * Set timer object
   */
  virtual void SetTimer(TimeMeasurement *timer) { timer_ = timer; }

  /**
   * Set quiet mode
   */
  virtual void SetQuietMode(bool quiet_mode) { quiet_mode_ = quiet_mode; }

  /**
   * Set the memory manager type
   */
  virtual void SetMemType(const std::string &mem_type) { mem_type_ = mem_type; }

  /**
   * Getter for the Work Group Size
   */
  uint32_t GetWorkGroupSize() const { return work_group_size; }

  /**
   * Getter for the number of compute units
   */
  uint32_t GetNumComputeUnits() const { return num_compute_units_; }

  /**
   * Setter for the compute unit count
   */
  void SetNumComputeUnits(uint32_t num_compute_units) {
    num_compute_units_ = num_compute_units;
  }
};

#endif  // SRC_COMMON_BENCHMARK_BENCHMARK_H_
