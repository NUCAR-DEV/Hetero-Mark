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

#include <string>
#include "gtest/gtest.h"
#include "src/common/benchmark/benchmark_runner.h"
#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/time_measurement/timer.h"
#include "src/common/time_measurement/time_keeper_impl.h"

TEST(BenchmarkRunner, run_benchmark) {
  class MockupTimeMeasurement : public TimeMeasurement {
   public:
    void Start() {}
    void End(std::initializer_list<const char *> catagories) {}
    void Summarize(std::ostream *ostream) {}
    double GetTime(const char *catagory) { return 0; }
  };

  class MockupBenchmark : public Benchmark {
   public:
    void Initialize() override { pass_ += "initialize,"; }
    void Run() override { pass_ += "run,"; }
    void Verify() override { pass_ += "verify,"; }
    void Summarize() override { pass_ += "summary,"; }
    void Cleanup() override { pass_ += "cleanup"; }
    const std::string &get_pass() const { return pass_; }
    void CleanPass() { pass_ = ""; }
   protected:
    std::string pass_ = "";
  };

  MockupBenchmark benchmark;
  MockupTimeMeasurement measurement;
  BenchmarkRunner runner(&benchmark, &measurement);

  runner.Run();

  const std::string &pass = benchmark.get_pass();
  EXPECT_STREQ("initialize,run,summary,cleanup", pass.c_str());

  benchmark.CleanPass();
  runner.set_verification_mode(true);
  runner.Run();
  const std::string &pass2 = benchmark.get_pass();
  EXPECT_STREQ("initialize,run,verify,summary,cleanup", pass2.c_str());
}
