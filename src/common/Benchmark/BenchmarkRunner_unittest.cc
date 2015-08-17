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
#include "hsa/common/Benchmark.h"
#include "hsa/common/TimeMeasurement.h"
#include "hsa/common/Timer.h"
#include "hsa/common/TimeKeeperImpl.h"
#include "gtest/gtest.h"

TEST(BenchmarkRunner, run_benchmark) {
  class MockupTimer : public Timer {
   public:
    double getTimeInSec() override {
      involkTime++;
      return static_cast<double>(involkTime);
    }
   protected:
    int involkTime = 0;
  };

  class MockupTimeMeasurement : public TimeMeasurement {
   public:
    MockupTimeMeasurement(std::unique_ptr<Timer> timer) {
      this->timer = std::move(timer);
      timeKeeper.reset(new TimeKeeperImpl(this->timer.get()));
    }
    void summarize(std::ostream *ostream) {}
  };

  class MockupBenchmark : public Benchmark {
   public:
    void initialize() override { pass += "initialize,"; }
    void run() override { pass += "run,"; }
    void verify() override { pass += "verify,"; }
    void summarize() override { pass += "summary,"; }
    void cleanUp() override { pass += "cleanup"; }
    const std::string &getPass() const { return pass; }
    void cleanPass() { pass = ""; }
   protected:
    std::string pass = "";
  };

  MockupBenchmark benchmark;
  std::unique_ptr<MockupTimer> timer = 
    std::unique_ptr<MockupTimer>(new MockupTimer);
  MockupTimeMeasurement measurement(std::move(timer));
  BenchmarkRunner runner(&benchmark, &measurement);

  runner.run();
  
  const std::string &pass = benchmark.getPass();
  EXPECT_STREQ("initialize,run,summary,cleanup", pass.c_str());

  benchmark.cleanPass();
  runner.setVerificationMode(true);
  runner.run();
  const std::string &pass2 = benchmark.getPass();
  EXPECT_STREQ("initialize,run,verify,summary,cleanup", pass2.c_str());
}
