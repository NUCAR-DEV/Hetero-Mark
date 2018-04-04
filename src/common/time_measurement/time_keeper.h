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

#ifndef SRC_COMMON_TIME_MEASUREMENT_TIME_KEEPER_H_
#define SRC_COMMON_TIME_MEASUREMENT_TIME_KEEPER_H_

#include <memory>
#include <string>
#include <utility>

/**
 * A TimeKeeper is responsible for keep track of how time is used
 */
class TimeKeeper {
 public:
  /**
   * The iterator that goes through all catagories
   */
  class Iterator {
   public:
    virtual ~Iterator() {}
    virtual bool HasNext() = 0;
    virtual std::pair<std::string, double> Next() = 0;
  };

  /**
   * Virtual destructor
   */
  virtual ~TimeKeeper() {}

  /**
   * Start the timer. Timer cannot be nested. That means the timer must be
   * ended before start again.
   */
  virtual void Start() = 0;

  /**
   * End the time and set the catagories that the time since timer start
   * belongs to.
   */
  virtual void End(std::initializer_list<const char *> catagories) = 0;

  /**
   * Get the iterator that walks through all catagories
   */
  virtual std::unique_ptr<Iterator> GetCatagoryIterator() = 0;

  /**
   * Get the time used in a catagory by the catagory's name
   */
  virtual double GetTime(const char *catagory_name) = 0;
};

#endif  // SRC_COMMON_TIME_MEASUREMENT_TIME_KEEPER_H_
