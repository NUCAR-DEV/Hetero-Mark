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

#include "src/common/command_line_option/option_parser.h"

#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "src/common/command_line_option/option_parser_impl.h"
#include "src/common/command_line_option/option_setting.h"
#include "src/common/command_line_option/option_setting_impl.h"

TEST(OptionParserImpl, parse) {
  // Setup environment
  ArgumentValueFactory argument_value_factory;
  auto option_setting =
      std::unique_ptr<OptionSetting>(new OptionSettingImpl("", ""));
  auto option_parser = std::unique_ptr<OptionParser>(
      new OptionParserImpl(option_setting.get(), &argument_value_factory));

  // Setup arguments that uses a short prompt
  auto arg1 = std::unique_ptr<Argument>(new Argument("name"));
  arg1->set_short_prompt("-n");
  arg1->set_type("string");
  option_setting->AddArgument(std::move(arg1));

  // Setup argument that uses a long prompt
  auto arg2 = std::unique_ptr<Argument>(new Argument("arg2"));
  arg2->set_long_prompt("--arg2");
  arg2->set_type("bool");
  option_setting->AddArgument(std::move(arg2));

  // Setup an argument with default value
  auto arg3 = std::unique_ptr<Argument>(new Argument("arg3"));
  arg3->set_long_prompt("--arg3");
  arg3->set_type("int32");
  arg3->set_default_value("1234");
  option_setting->AddArgument(std::move(arg3));

  // Configure user input
  int argc = 4;
  const char *argv[] = {"run", "--arg2", "-n", "name"};

  // Parse
  option_parser->Parse(argc, argv);

  // Check result
  ArgumentValue *value1 = option_parser->GetValue("name");
  ASSERT_TRUE(value1 != nullptr);
  EXPECT_STREQ("name", value1->AsString().c_str());
  ArgumentValue *value2 = option_parser->GetValue("arg2");
  ASSERT_TRUE(value2 != nullptr);
  EXPECT_STREQ("true", value2->AsString().c_str());
  ArgumentValue *value3 = option_parser->GetValue("arg3");
  ASSERT_TRUE(value3 != nullptr);
  EXPECT_EQ(1234, value3->AsInt32());
}
