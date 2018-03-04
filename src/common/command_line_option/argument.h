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

#ifndef SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_H_
#define SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_H_

#include <string>

/**
 * An argument is a value that can be set via the command line option
 */
class Argument {
 protected:
  // Name of the argument
  std::string name_;

  // Short prompt is the leading word of the argument, starting with a single
  // dash. It should only has error. For example "-h" for help information
  std::string short_prompt_;

  // Long prompt is the leading word of the argument, starting with two dashes.
  // It is usually a word. For example, "--help" for help information
  std::string long_prompt_;

  // Description of the argument
  std::string description_;

  // Default value
  std::string default_value_;

  // Type of the argument. This field does not do any validation. It is only
  // used as part of help messages.
  std::string type_;

 public:
  /**
   * Constructor
   */
  explicit Argument(const char *name) : name_(name) {}

  virtual ~Argument() {}

  /**
   * Get the name of the argument
   */
  virtual const std::string get_name() { return name_; }

  /**
   * Set short prompt
   */
  virtual void set_short_prompt(const char *short_prompt) {
    short_prompt_ = short_prompt;
  }

  /**
   * Get the short prompt format
   */
  virtual const std::string get_short_prompt() { return short_prompt_; }

  /**
   * Set long prompt
   */
  virtual void set_long_prompt(const char *long_prompt) {
    long_prompt_ = long_prompt;
  }

  /**
   * Get the long prompt format
   */
  virtual const std::string get_long_prompt() { return long_prompt_; }

  /**
   * Set type information
   */
  virtual void set_type(const char *type) { type_ = type; }

  /**
   * Get the type string
   */
  virtual const std::string get_type() { return type_; }

  /**
   * Set the default value
   */
  virtual void set_default_value(const char *default_value) {
    default_value_ = default_value;
  }

  /**
   * Get the default value
   */
  virtual const std::string get_default_value() { return default_value_; }

  /**
   * Set the description
   */
  virtual void set_description(const char *description) {
    description_ = description;
  }

  /**
   * Get the description
   */
  virtual const std::string get_description() { return description_; }
};

#endif  // SRC_COMMON_COMMAND_LINE_OPTION_ARGUMENT_H_
