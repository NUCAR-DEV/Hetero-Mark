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

#ifndef SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_VALUE_H_
#define SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_VALUE_H_

#include <string>

class ArgumentValue {
 protected:
  // The value of the argument. It is always stored as an string. Users need
  // to convert is explicitly into desired types with as<Type> functions
  std::string value_;

 public:
  /**
   * Constructor. The value of the newly created instance will be set to
   * empty at the beginning
   */
  ArgumentValue() : value_() {
  }

  /**
   * Set the value in string format
   */
  virtual void set_value(const char *value) { value_ = value; }

  /**
   * Return the value in type of string
   */
  virtual const std::string AsString() {
    return value_;
  }

  /**
   * Return the value in type of int32_t
   * This function may throw error. The caller should catch the error
   */
  virtual int32_t AsInt32() {
    int32_t integer;
    integer = stoi(value_);
    return integer;
  }

  /**
   * Return the value in type of uint32_t
   */
  virtual uint32_t AsUInt32() {
    uint32_t integer;
    integer = stoi(value_);
    return integer;
  }

  virtual bool AsBool() {
    if (value_ == "true") {
      return true;
    } else if (value_ == "false") {
      return false;
    } else {
      throw std::runtime_error(std::string("Value ") + value_ + " cannot be"
          "interpreted as bool");
    }
  }

  virtual int64_t AsInt64() {
    int64_t integer;
    integer = stol(value_);
    return integer;
  }

  virtual uint64_t AsUInt64() {
    uint64_t integer;
    integer = stoul(value_);
    return integer;
  }
};

#endif  // SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_VALUE_H_
