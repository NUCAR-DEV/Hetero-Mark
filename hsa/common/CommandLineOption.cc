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

#include <stdexcept>
#include <string>
#include "hsa/common/CommandLineOption.h"
#include "hsa/common/OptionSettingImpl.h"
#include "hsa/common/OptionParserImpl.h"
#include "hsa/common/OptionSettingHelpPrinter.h"
#include "hsa/common/Argument.h"

CommandLineOption::CommandLineOption(const char *name,
    const char *description) {
  optionSetting.reset(new OptionSettingImpl());
}

void CommandLineOption::addArgument(const char *name,
      const char *type, const char *defaultValue,
      const char *shortPrompt, const char *longPrompt,
      const char *description) {
  auto argument = std::unique_ptr<Argument>(new Argument(name));
  argument->setType(type);
  argument->setDefaultValue(defaultValue);
  argument->setShortPrompt(shortPrompt);
  argument->setLongPrompt(longPrompt);
  argument->setDescription(description);
  optionSetting->addArgument(std::move(argument));
}

void CommandLineOption::parse(int argc, const char **argv) {
  ArgumentValueFactory argumentValueFactory;
  optionParser.reset(new OptionParserImpl(optionSetting.get(),
        &argumentValueFactory));
  optionParser->parse(argc, argv);
}

void CommandLineOption::help(std::ostream *ostream) {
  OptionSettingHelpPrinter printer(optionSetting.get());
  printer.print(ostream);
}

ArgumentValue *CommandLineOption::getArgumentValue(const char *name) {
  // Check if the arguments have been parsed
  if (!optionParser.get()) {
    throw std::runtime_error("Command line argument not parsed. Call parse() "
        "before getArgumentValue");
  }

  // Try to get argument value
  ArgumentValue *argumentValue = optionParser->getValue(name);
  if (!argumentValue) {
    throw std::runtime_error(std::string("Argument ") + name + " not found");
  }

  // Return argument value
  return argumentValue;
}
