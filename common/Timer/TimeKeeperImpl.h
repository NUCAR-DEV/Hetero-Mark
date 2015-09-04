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

#ifndef SRC_COMMON_TIMER_TIMEKEEPERIMPL_H_
#define SRC_COMMON_TIMER_TIMEKEEPERIMPL_H_

#include <string>
#include <utility>
#include <map>
#include "src/common/Timer/TimeKeeper.h"
#include "src/common/Timer/Timer.h"

class TimeKeeperImpl : public TimeKeeper {
 public:
  /**
   * The iterator that goes through all catagories
   */
  class Iterator : public TimeKeeper::Iterator{
   public:
    Iterator(std::map<std::string, double>::iterator begin,
        std::map<std::string, double>::iterator end);

    bool hasNext() override;;

    std::pair<std::string, double> next() override;
   private:
    std::map<std::string, double>::iterator begin;
    std::map<std::string, double>::iterator iterator;
    std::map<std::string, double>::iterator end;
  };

  /**
   * Constructor
   *
   * @param: timer The timer object is depends on
   */
  explicit TimeKeeperImpl(Timer *timer);

  /**
   * Destructor
   */
  virtual ~TimeKeeperImpl() {}

  /**
   * Start the timer. Timer cannot be nested. That means the timer must be 
   * ended before start again.
   */
  void start() override;

  /**
   * End the time and set the catagories that the time since timer start 
   * belongs to.
   */
  void end(std::initializer_list<const char *> catagories) override;

  /**
   * Get the iterator that walks through all catagories
   */
  std::unique_ptr<TimeKeeper::Iterator> getCatagoryIterator() override;

  /**
   * Get time used in a catagory by the catagory's name
   */
  double getTime(const char *catagory_name) override;

 protected:
  std::map<std::string, double> timeCatagories;
  Timer *timer;
  double startTime = 0;
};

#endif  // SRC_COMMON_TIMER_TIMEKEEPERIMPL_H_
