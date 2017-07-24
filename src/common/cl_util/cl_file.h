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
 * Author: Xiang Gong (xgong@ece.neu.edu)
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
#ifndef SRC_COMMON_CL_UTIL_CL_FILE_H_
#define SRC_COMMON_CL_UTIL_CL_FILE_H_

#include <stdio.h>
#include <fstream>
#include <memory>
#include <string>

namespace clHelper {

class clFile {
 private:
  // Instance of the singleton
  static std::unique_ptr<clFile> instance;

  // Private constructor for singleton
  clFile() : sourceCode("") {}

  // Disable copy constructor
  clFile(const clFile &) = delete;

  // Disable operator=
  clFile &operator=(const clFile &) = delete;

  // source code of the CL program
  std::string sourceCode;

 public:
  // Get singleton
  static clFile *getInstance();

  ~clFile() {}

  // Getters
  const std::string &getSource() const { return sourceCode; }

  const char *getSourceChar() const { return sourceCode.c_str(); }

  // Read file
  bool open(const char *fileName);
};

}  // namespace clHelper

#endif  // SRC_COMMON_CL_UTIL_CL_FILE_H_
