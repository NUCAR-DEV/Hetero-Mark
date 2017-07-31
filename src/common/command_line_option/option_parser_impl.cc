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

#include "src/common/command_line_option/option_parser_impl.h"

#include <iostream>
#include <stdexcept>
#include <utility>

void OptionParserImpl::Parse(int argc, const char **argv) {
  // First of all, set all argument with default values
  auto iterator = option_setting_->GetIterator();
  while (iterator->HasNext()) {
    Argument *argument = iterator->Next();
    std::string default_value = argument->get_default_value();
    auto argument_value =
        argument_value_factory_->ProduceArgumentValue(default_value.c_str());
    argument_values_.emplace(argument->get_name(), std::move(argument_value));
  }

  // Skip argv[0], because it is the executable name
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    // Each iteration must start with a prompt, either long or short.
    // Therefore, the length must greater than 2
    if (arg.length() == 0) {
      // throw std::runtime_error(std::string("Argument size is 0"));
      // Empty args, skip
      continue;
    } else if (arg.length() == 1) {
      throw std::invalid_argument((std::string("Invalid argument ") + arg));
    }

    // Check if it is long or shor prompt
    bool is_short_prompt = false;
    bool is_long_prompt = false;
    if (arg[0] == '-' && arg[1] == '-')
      is_long_prompt = true;
    else if (arg[0] == '-')
      is_short_prompt = true;

    // According to the prompt, find the argument
    // FIXME:Yifan Move this part of code to another function
    Argument *argument = nullptr;
    if (is_short_prompt) {
      auto it = option_setting_->GetIterator();
      while (it->HasNext()) {
        Argument *search_argument = it->Next();
        if (search_argument->get_short_prompt() == arg) {
          argument = search_argument;
          break;
        }
      }
    } else if (is_long_prompt) {
      auto it = option_setting_->GetIterator();
      while (it->HasNext()) {
        Argument *search_argument = it->Next();
        if (search_argument->get_long_prompt() == arg) {
          argument = search_argument;
          break;
        }
      }
    }

    // Throw error if argument is not found
    if (!argument) {
      throw std::invalid_argument(std::string("Invalid argument ") + arg);
    }

    // Bool type argument is treated differently
    if (argument->get_type() == "bool") {
      // Create argument value
      auto argument_value = argument_values_.find(argument->get_name());
      argument_value->second->set_value("true");
    } else {
      // Non-bool type prompt cannot be the last argument
      if (i == argc - 1) {
        throw std::runtime_error("Argument " + arg +
                                 "must followed by its "
                                 "value");
      }

      // Consumes one more argument
      // FIXME: validate argument type
      std::string arg2(argv[i + 1]);
      i++;
      auto argument_value = argument_values_.find(argument->get_name());
      argument_value->second->set_value(arg2.c_str());
    }
  }
}

ArgumentValue *OptionParserImpl::GetValue(const char *name) {
  // Find argument value
  auto it = argument_values_.find(name);

  // Argument value not exist, return nullptr
  if (it == argument_values_.end()) {
    return nullptr;
  }

  // Argument value found, return the pointer to the value object
  return it->second.get();
}
