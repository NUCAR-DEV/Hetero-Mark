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

#ifndef SRC_COMMON_COMMAND_LINE_OPTION_OPTION_SETTING_IMPL_H_
#define SRC_COMMON_COMMAND_LINE_OPTION_OPTION_SETTING_IMPL_H_

#include <map>
#include <memory>
#include <string>

#include "src/common/command_line_option/option_setting.h"

class OptionSettingImpl : public OptionSetting {
 public:
  /**
   * The iterator for arguments
   */
  class Iterator : public OptionSetting::Iterator {
   public:
    Iterator(std::map<std::string, std::unique_ptr<Argument>>::iterator begin,
             std::map<std::string, std::unique_ptr<Argument>>::iterator end)
        : begin_(begin), iterator_(begin), end_(end) {}
    bool HasNext() override;
    Argument *Next() override;

   private:
    std::map<std::string, std::unique_ptr<Argument>>::iterator begin_;
    std::map<std::string, std::unique_ptr<Argument>>::iterator iterator_;
    std::map<std::string, std::unique_ptr<Argument>>::iterator end_;
  };

  OptionSettingImpl() {}
  ~OptionSettingImpl() {}

  /**
   * Add an argument
   */
  void AddArgument(std::unique_ptr<Argument> argument) override;

  /**
   * Get the argument iterator
   */
  std::unique_ptr<OptionSetting::Iterator> GetIterator() override;

  const std::string GetProgramName() override { return name_; }
  void SetProgramName(const char *name) override { name_ = std::string(name); }

  const std::string GetProgramDescription() override { return description_; }
  void SetProgramDescription(const char *description) override {
    description_ = std::string(description);
  }

 protected:
  std::map<std::string, std::unique_ptr<Argument>> arguments_;
  std::string name_;
  std::string description_;
};

#endif  // SRC_COMMON_COMMAND_LINE_OPTION_OPTION_SETTING_IMPL_H_
