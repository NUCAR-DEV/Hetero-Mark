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

#include "src/common/time_measurement/time_keeper.h"
#include <gtest/gtest.h>
#include "src/common/time_measurement/time_keeper_impl.h"
#include "src/common/time_measurement/timer.h"

TEST(TimeKeeperImpl, keep_time) {
  class MockupTimer : public Timer {
   public:
    double GetTimeInSec() override {
      involkTime++;
      return static_cast<double>(involkTime);
    }

   private:
    int involkTime = 0;
  };

  // Create environment
  MockupTimer timer;
  TimeKeeperImpl time_keeper(&timer);

  // Count
  time_keeper.Start();
  time_keeper.End({"catagoryA"});
  time_keeper.Start();
  time_keeper.End({"catagoryA", "catagoryB"});

  // Get Iterator
  auto it = time_keeper.GetCatagoryIterator();

  // Assert result
  auto pair = it->Next();
  EXPECT_STREQ("catagoryA", pair.first.c_str());
  EXPECT_DOUBLE_EQ(2.0, pair.second);
  pair = it->Next();
  EXPECT_STREQ("catagoryB", pair.first.c_str());
  EXPECT_DOUBLE_EQ(1.0, pair.second);
}
