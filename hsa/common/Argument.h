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

#ifndef HSA_COMMON_ARGUMENT_H_
#define HSA_COMMON_ARGUMENT_H_

#include <string>

/**
 * An argument is a value that can be set via the command line option
 */
class Argument {
 protected:
  // Name of the argument
  std::string name;

  // Short prompt is the leading word of the argument, starting with a single
  // dash. It should only has error. For example "-h" for help information
  std::string shortPrompt;

  // Long prompt is the leading word of the argument, starting with two dashes.
  // It is usually a word. For example, "--help" for help information
  std::string longPrompt;

  // Description of the argument
  std::string description;

  // Default value
  std::string defaultValue;

  // Type of the argument. This field does not do any validation. It is only
  // used as part of help messages.
  std::string type;

 public:
  /**
   * Constructor 
   */
  explicit Argument(const char *name) : name(name) {}

  /**
   * Get the name of the argument
   */
  virtual const std::string getName() { return name; }

  /**
   * Set short prompt
   */
  virtual void setShortPrompt(const char *shortPrompt) {
    this->shortPrompt = shortPrompt;
  }

  /**
   * Get the short prompt format
   */
  virtual const std::string getShortPrompt() { return shortPrompt; }

  /**
   * Set long prompt
   */
  virtual void setLongPrompt(const char *longPrompt) {
    this->longPrompt = longPrompt;
  }

  /**
   * Get the long prompt format
   */
  virtual const std::string getLongPrompt() { return longPrompt; }

  /**
   * Set type information
   */
  virtual void setType(const char *type) {
    this->type = type;
  }

  /**
   * Get the type string
   */
  virtual const std::string getType() { return type; }

  /**
   * Set the default value
   */
  virtual void setDefaultValue(const char *defaultValue) {
    this->defaultValue = defaultValue;
  }

  /**
   * Get the default value
   */
  virtual const std::string getDefaultValue() { return defaultValue; }

  /**
   * Set the description
   */
  virtual void setDescription(const char *description) {
    this->description = description;
  }

  /**
   * Get the description
   */
  virtual const std::string getDescription() { return description; }
};

#endif  // HSA_COMMON_ARGUMENT_H_
