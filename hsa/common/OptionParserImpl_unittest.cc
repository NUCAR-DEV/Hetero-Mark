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

#include <memory>
#include "hsa/common/OptionSetting.h"
#include "hsa/common/OptionSettingImpl.h"
#include "hsa/common/OptionParser.h"
#include "hsa/common/OptionParserImpl.h"
#include "gtest/gtest.h"

TEST(OptionParserImpl, parse) {
  // Setup environment
  ArgumentValueFactory argumentValueFactory;
  auto optionSetting = std::unique_ptr<OptionSetting>(new OptionSettingImpl());
  auto optionParser = std::unique_ptr<OptionParser>(new OptionParserImpl(
        optionSetting.get(), &argumentValueFactory));

  // Setup arguments that uses a short prompt
  auto arg1 = std::unique_ptr<Argument>(new Argument("name"));
  arg1->setShortPrompt("-n");
  arg1->setType("string");
  optionSetting->addArgument(std::move(arg1));

  // Setup argument that uses a long prompt
  auto arg2 = std::unique_ptr<Argument>(new Argument("arg2"));
  arg2->setLongPrompt("--arg2");
  arg2->setType("bool");
  optionSetting->addArgument(std::move(arg2));

  // Setup an argument with default value
  auto arg3 = std::unique_ptr<Argument>(new Argument("arg3"));
  arg3->setLongPrompt("--arg3");
  arg3->setType("int32");
  arg3->setDefaultValue("1234");
  optionSetting->addArgument(std::move(arg3));

  // Configure user input
  int argc = 4;
  const char *argv[] = {"run", "-n", "name", "--arg2"};

  // Parse
  optionParser->parse(argc, argv);

  // Check result
  ArgumentValue *value1 = optionParser->getValue("name");
  ASSERT_TRUE(value1!=nullptr);
  EXPECT_STREQ("name", value1->asString().c_str());
  ArgumentValue *value2 = optionParser->getValue("arg2");
  ASSERT_TRUE(value2!=nullptr);
  EXPECT_STREQ("true", value2->asString().c_str());
  ArgumentValue *value3 = optionParser->getValue("arg3");
  ASSERT_TRUE(value3!=nullptr);
  EXPECT_EQ(1234, value3->asInt32());
}
