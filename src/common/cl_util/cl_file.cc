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

#include "src/common/cl_util/cl_file.h"

namespace clHelper {

// Singleton instance
std::unique_ptr<clFile> clFile::instance;

clFile *clFile::getInstance() {
  // Instance already exists
  if (instance.get()) return instance.get();

  // Create instance
  instance.reset(new clFile());
  return instance.get();
}

bool clFile::open(const char *fileName) {
  size_t size;
  char *str;

  // Open file stream
  std::fstream f(fileName, (std::fstream::in | std::fstream::binary));

  // Check if we have opened file stream
  if (f.is_open()) {
    size_t sizeFile;

    // Find the stream size
    f.seekg(0, std::fstream::end);
    size = sizeFile = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size + 1];
    if (!str) {
      f.close();
      return false;
    }

    // Read file
    f.read(str, sizeFile);
    f.close();
    str[size] = '\0';
    sourceCode = str;
    delete[] str;

    return true;
  }

  return false;
}

}  // namespace clHelper
