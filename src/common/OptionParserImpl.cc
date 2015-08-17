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

#include <iostream>
#include <stdexcept>

#include "hsa/common/OptionParserImpl.h"

void OptionParserImpl::parse(int argc, const char **argv) {
  // First of all, set all argument with default values
  auto iterator = optionSetting->getIterator();
  while (iterator->hasNext()) {
    Argument *argument = iterator->next();
    std::string defaultValue = argument->getDefaultValue();
    auto argumentValue = argumentValueFactory->produceArgumentValue(
          defaultValue.c_str());
    argumentValues.emplace(argument->getName(), std::move(argumentValue));
  }

  // Skip argv[0], because it is the executable name
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    // Each iteration must start with a prompt, either long or short.
    // Therefore, the length must greater than 2
    if (arg.length() == 0) {
      throw std::runtime_error(std::string("Argument size is 0"));
    } else if (arg.length() == 1) {
      throw std::invalid_argument((std::string("Invalid argument ") +
            arg));
    }

    // Check if it is long or shor prompt
    bool isShortPrompt = false;
    bool isLongPrompt = false;
    if (arg[0] == '-' && arg[1] == '-') isLongPrompt = true;
    else if (arg[0] == '-') isShortPrompt = true;

    // According to the prompt, find the argument
    // FIXME Move this part of code to another function
    Argument *argument = nullptr;
    if (isShortPrompt) {
      auto it = optionSetting->getIterator();
      while (it->hasNext()) {
        Argument *searchArgument = it->next();
        if (searchArgument->getShortPrompt() == arg) {
          argument = searchArgument;
          break;
        }
      }
    } else if (isLongPrompt) {
      auto it = optionSetting->getIterator();
      while (it->hasNext()) {
        Argument *searchArgument = it->next();
        if (searchArgument->getLongPrompt() == arg) {
          argument = searchArgument;
          break;
        }
      }
    }

    // Throw error if argument is not found
    if (!argument) {
      throw std::invalid_argument(std::string("Invalid argument ") + arg);
    }

    // Bool type argument is treated differently
    if (argument->getType() == "bool") {
      // Create argument value
      auto argumentValue = argumentValues.find(argument->getName());
      argumentValue->second->setValue("true");
      return;
    } else {
      // Non-bool type prompt cannot be the last argument
      if (i == argc - 1) {
        throw std::runtime_error("Argument " + arg + "must followed by its "
            "value");
      }

      // Consumes one more argument
      // FIXME: validate argument type
      std::string arg2(argv[i + 1]);
      i++;
      auto argumentValue = argumentValues.find(argument->getName());
      argumentValue->second->setValue(arg2.c_str());
    }
  }
}

ArgumentValue *OptionParserImpl::getValue(const char *name) {
  // Find argument value
  auto it = argumentValues.find(name);

  // Argument value not exist, return nullptr
  if (it == argumentValues.end()) {
    return nullptr;
  }

  // Argument value found, return the pointer to the value object
  return it->second.get();
}
