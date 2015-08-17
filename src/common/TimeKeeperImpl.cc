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

#include <stdexcept>
#include <memory>
#include "hsa/common/TimeKeeperImpl.h"

TimeKeeperImpl::Iterator::Iterator(
    std::map<std::string, double>::iterator begin,
    std::map<std::string, double>::iterator end) :
  begin(begin),
  iterator(begin),
  end(end) {}

bool TimeKeeperImpl::Iterator::hasNext() {
  if (iterator == end) {
    return false;
  } else {
    return true;
  }
}

std::pair<std::string, double> TimeKeeperImpl::Iterator::next() {
  std::pair<std::string, double> pair(iterator->first, iterator->second);
  iterator++;
  return pair;
}

TimeKeeperImpl::TimeKeeperImpl(Timer *timer) : timer(timer) {
}

void TimeKeeperImpl::start() {
  if (startTime != 0) {
    throw std::runtime_error("Timer is already runnin.");
  }
  startTime = timer->getTimeInSec();
}

void TimeKeeperImpl::end(std::initializer_list<const char *> catagoryNames) {
  if (startTime == 0) {
    throw std::runtime_error("Timer has not been started.");
  }
  double endTime = timer->getTimeInSec();
  double timeDifference = endTime - startTime;
  startTime = 0;

  // Accumulate the time into catagories
  for (auto catagoryName : catagoryNames) {
    auto catagory = timeCatagories.find(catagoryName);
    if (catagory == timeCatagories.end()) {
      timeCatagories.emplace(catagoryName, 0);
      catagory = timeCatagories.find(catagoryName);
    }
    catagory->second += timeDifference;
  }
}

std::unique_ptr<TimeKeeper::Iterator> TimeKeeperImpl::getCatagoryIterator() {
  Iterator *it = new Iterator(timeCatagories.begin(), timeCatagories.end());
  std::unique_ptr<TimeKeeper::Iterator> it_unique(it);
  return std::move(it_unique);
}

double TimeKeeperImpl::getTime(const char *catagory_name) {
  auto catagory = timeCatagories.find(catagory_name);
  if (catagory == timeCatagories.end()) {
    return 0;
  } else {
    return catagory->second;}
}
