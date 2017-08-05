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

#ifndef SRC_COMMON_COMMAND_LINE_OPTION_COMMAND_LINE_OPTION_H_
#define SRC_COMMON_COMMAND_LINE_OPTION_COMMAND_LINE_OPTION_H_

#include <iostream>
#include <memory>

#include "src/common/command_line_option/argument_value.h"
#include "src/common/command_line_option/option_parser.h"
#include "src/common/command_line_option/option_setting.h"

/**
 * A CommandLineOption is a facade for the command line argument parsing
 * system.
 */
class CommandLineOption {
 public:
  CommandLineOption();

  void SetBenchmarkName(const char *name) {
    option_setting_->SetProgramName(name);
  }

  void SetDescription(const char *description) {
    option_setting_->SetProgramDescription(description);
  }

  void AddArgument(const char *name, const char *type, const char *defaultValue,
                   const char *shortPrompt, const char *longPrompt,
                   const char *description);
  void Parse(int argc, const char **argv);
  void Help(std::ostream *ostream = &std::cout);
  ArgumentValue *GetArgumentValue(const char *name);

 protected:
  std::unique_ptr<OptionSetting> option_setting_;
  std::unique_ptr<OptionParser> option_parser_;
};

#endif  // SRC_COMMON_COMMAND_LINE_OPTION_COMMAND_LINE_OPTION_H_
