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

#include "src/common/time_measurement/time_keeper_impl.h"
#include <memory>
#include <stdexcept>

TimeKeeperImpl::Iterator::Iterator(
    std::map<std::string, double>::iterator begin,
    std::map<std::string, double>::iterator end)
    : begin_(begin), iterator_(begin), end_(end) {}

bool TimeKeeperImpl::Iterator::HasNext() {
  if (iterator_ == end_) {
    return false;
  } else {
    return true;
  }
}

std::pair<std::string, double> TimeKeeperImpl::Iterator::Next() {
  std::pair<std::string, double> pair(iterator_->first, iterator_->second);
  iterator_++;
  return pair;
}

TimeKeeperImpl::TimeKeeperImpl(Timer *timer) : timer_(timer) {}

void TimeKeeperImpl::Start() {
  if (start_time_ > 0) {
    throw std::runtime_error("Timer is already running.");
  }
  start_time_ = timer_->GetTimeInSec();
}

void TimeKeeperImpl::End(std::initializer_list<const char *> catagory_names) {
  if (start_time_ < 0) {
    throw std::runtime_error("Timer has not been started.");
  }
  double end_time = timer_->GetTimeInSec();
  double time_difference = end_time - start_time_;
  start_time_ = -1;

  // Accumulate the time into catagories
  for (auto catagory_name : catagory_names) {
    auto catagory = time_catagories_.find(catagory_name);
    if (catagory == time_catagories_.end()) {
      time_catagories_.emplace(catagory_name, 0);
      catagory = time_catagories_.find(catagory_name);
    }
    catagory->second += time_difference;
  }
}

std::unique_ptr<TimeKeeper::Iterator> TimeKeeperImpl::GetCatagoryIterator() {
  Iterator *it = new Iterator(time_catagories_.begin(), time_catagories_.end());
  std::unique_ptr<TimeKeeper::Iterator> it_unique(it);
  return it_unique;
}

double TimeKeeperImpl::GetTime(const char *catagory_name) {
  auto catagory = time_catagories_.find(catagory_name);
  if (catagory == time_catagories_.end()) {
    return 0;
  } else {
    return catagory->second;
  }
}
