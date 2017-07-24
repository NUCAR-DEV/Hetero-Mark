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

#ifndef SRC_COMMON_CL_UTIL_CL_PROFILER_H_
#define SRC_COMMON_CL_UTIL_CL_PROFILER_H_

#include <CL/cl.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/common/cl_util/cl_runtime.h"

namespace clHelper {

class clProfilerMeta {
  std::string name;
  double totalTime;

  int limit;

  std::vector<std::unique_ptr<std::pair<double, double>>> timeTable;

 public:
  explicit clProfilerMeta(std::string nm);
  ~clProfilerMeta();

  /// Getters
  const std::string getName() const { return name; }

  const double getTotalTime() const { return totalTime; }

  /// Record a profiling information
  void insert(double st, double ed);

  /// Operator \c << invoking the function Dump on an output stream
  friend std::ostream &operator<<(std::ostream &os,
                                  const clProfilerMeta &clProfMeta) {
    clProfMeta.Dump(&os);
    return os;
  }

  void Dump(std::ostream *os) const;
};

class clProfiler {
  // Instance of the singleton
  static std::unique_ptr<clProfiler> instance;

  // Private constructor for singleton
  clProfiler();

  // Contains profiling data
  std::vector<std::unique_ptr<clProfilerMeta>> profilingData;

  // String length
  size_t strLen;

 public:
  ~clProfiler();

  // Get singleton
  static clProfiler *getInstance();

  // Get number of record
  int getNumRecord() const { return profilingData.size(); }

  // Dump kernel profiling time
  void getExecTime(std::string name = "");

  // Add profiling info
  void addExecTime(std::string name, double st, double ed);

  // Set max string length
  void setStringLen(size_t strLen) { this->strLen = strLen; }
};

cl_int clProfileNDRangeKernel(cl_command_queue cmdQ, cl_kernel kernel,
                              cl_uint wd, const size_t *glbOs,
                              const size_t *glbSz, const size_t *locSz,
                              cl_uint numEvt, const cl_event *evtLst,
                              cl_event *evt);

cl_int clTimeNDRangeKernel(cl_command_queue cmdQ, cl_kernel kernel, cl_uint wd,
                           const size_t *glbOs, const size_t *glbSz,
                           const size_t *locSz, cl_uint numEvt,
                           const cl_event *evtLst, cl_event *evt);
}  // namespace clHelper

#endif  // SRC_COMMON_CL_UTIL_CL_PROFILER_H_
