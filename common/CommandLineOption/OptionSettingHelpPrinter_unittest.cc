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

#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "src/common/CommandLineOption/OptionSettingHelpPrinter.h"
#include "src/common/CommandLineOption/OptionSetting.h"

TEST(OptionSettingHelpPrinter, print) {
  class MockupArgument : public Argument {
   public:
    explicit MockupArgument(const char *name) : Argument(name) {}
    const std::string getShortPrompt() { return "-n"; }
    const std::string getLongPrompt() { return "--name"; }
    const std::string getType() { return "string"; }
    const std::string getDescription() { return "This is a description"; }
    const std::string getDefaultValue() { return "default"; }
  };

  class MockupOptionSetting : public OptionSetting {
    class MockupIterator : public OptionSetting::Iterator {
     public:
      MockupIterator() {
        arg1.reset(new MockupArgument("arg1"));
        arg2.reset(new MockupArgument("arg2"));
      }
      bool hasNext() {
        if (index == 2) return false;
        return true;
      }
      Argument *next() {
        index++;
        if (index == 1) return arg1.get();
        if (index == 2) return arg2.get();
        return nullptr;
      }
     private:
      int index = 0;
      std::unique_ptr<Argument> arg1;
      std::unique_ptr<Argument> arg2;
    };

    std::unique_ptr<OptionSetting::Iterator> getIterator() override {
      OptionSetting::Iterator *it = new MockupOptionSetting::MockupIterator();
      std::unique_ptr<OptionSetting::Iterator> it_unique(it);
      return std::move(it_unique);
    }

    void addArgument(std::unique_ptr<Argument> argument) override {};

    const std::string getProgramName() override {
      return "Test program";
    }

    const std::string getProgramDescription() override {
      return "Test description";
    }
  };

  MockupOptionSetting mockupOptionSetting;
  std::stringstream stringstream;
  OptionSettingHelpPrinter printer(&mockupOptionSetting);
  printer.print(&stringstream);

  std::string result = stringstream.str();
  EXPECT_STREQ(
      "\nTest program\n"
      "Test description\n\n"
      "arg1[string]: -n --name (default = default)\n"
      "  This is a description\n\n"
      "arg2[string]: -n --name (default = default)\n"
      "  This is a description\n\n",
      result.c_str());
}
