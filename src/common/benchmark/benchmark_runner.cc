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
 *   its contributors may be used to Endorse or promote products derived
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

void BenchmarkRunner::Run() {
  time_measurement_->Start();
  benchmark_->Initialize();
  time_measurement_->End({"Initialize"});

  time_measurement_->Start();
  for (uint32_t i = 0; i < warm_up_time_; i++) {
    benchmark_->Run();
  }
  time_measurement_->End({"WarmUp"});

  time_measurement_->Start();
  for (uint32_t i = 0; i < repeat_time_; i++) {
    benchmark_->Run();
  }
  time_measurement_->End({"Run"});

  if (verification_mode_) {
    time_measurement_->Start();
    benchmark_->Verify();
    time_measurement_->End({"Verify"});
  }

  if (!quiet_mode_) {
    time_measurement_->Start();
    benchmark_->Summarize();
    time_measurement_->End({"Summarize"});
  }

  time_measurement_->Start();
  benchmark_->Cleanup();
  time_measurement_->End({"Cleanup"});

  if (timing_mode_) {
    Summarize(&std::cerr);
  }
}

void BenchmarkRunner::Summarize(std::ostream *ostream) {
  time_measurement_->Summarize(ostream);
}
