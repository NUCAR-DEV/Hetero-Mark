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

#ifndef SRC_COMMON_TIME_MEASUREMENT_CPU_GPU_ACTIVITY_LOGGER_
#define SRC_COMMON_TIME_MEASUREMENT_CPU_GPU_ACTIVITY_LOGGER_

#include <memory>
#include <iostream>

#include "timer.h"
#include "timer_impl.h"

class CPUGPUActivityLogger {
  std::unique_ptr<Timer> timer_;
  double prev_time_;
  bool first_run_ = true;

  double cpu_ = 0.0;
  double gpu_ = 0.0;
  double both_ = 0.0;

  int cpu_instance_ = 0;
  int gpu_instance_ = 0;

  void AddTime() {
    double now = timer_->GetTimeInSec();
    if (first_run_) {
      first_run_ = false;
      prev_time_ = now;
      return;
    }

    double duration = now - prev_time_;
    prev_time_ = now;
    if (cpu_instance_ > 0 && gpu_instance_ > 0) {
      both_ += duration;
    } else if (cpu_instance_ > 0) {
      cpu_ += duration;
    } else if (gpu_instance_ > 0) {
      gpu_ += duration;
    }
  }

 public:
  CPUGPUActivityLogger() {
    timer_.reset(new TimerImpl());
  }

  ~CPUGPUActivityLogger() {
    if (cpu_instance_ > 0) {
      std::cerr << "Warning: CPU activity not properly closed.";
    }
    if (gpu_instance_ > 0) {
      std::cerr << "Warning: GPU activity not properly closed.";
    }
  }

  void CPUOn() {
    AddTime();
    cpu_instance_++;
  }

  void CPUOff() {
    AddTime();
    cpu_instance_--;
  }
  
  void GPUOn() {
    AddTime();
    gpu_instance_++;
  }

  void GPUOff() {
    AddTime();
    gpu_instance_--;
  }

  void Summarize() {
    std::cerr << "CPU: " << cpu_ 
              << ", GPU: " << gpu_ 
              << ", both:" << both_
              << "\n";
  }
};

#endif  // SRC_COMMON_TIME_MEASUREMENT_CPU_GPU_ACTIVITY_LOGGER_

