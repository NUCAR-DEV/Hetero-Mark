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

#include "src/common/cl_util/cl_profiler.h"
#include <CL/cl.h>

namespace clHelper {

clProfilerMeta::clProfilerMeta(std::string nm)
    : name(nm), totalTime(0.0f), limit(10) {}

clProfilerMeta::~clProfilerMeta() {}

void clProfilerMeta::insert(double st, double ed) {
  timeTable.emplace_back(new std::pair<double, double>(st, ed));
  totalTime += (ed - st);
}

void clProfilerMeta::Dump(std::ostream *os) const {
  (*os) << "\t" << name << " : " << totalTime << " ms" << std::endl;

  // Only dump detailed data when size is not too large
  int count = 0;
  for (auto &elem : timeTable) {
    double st = elem.get()->first;
    double ed = elem.get()->second;
    double lt = ed - st;
    (*os) << "\t\t" << std::setprecision(32) << st << " - "
          << std::setprecision(32) << ed << ": " << std::setprecision(20) << lt
          << " ns, #" << count << std::endl;

    count++;
    if (count > limit) {
      (*os) << "\t\t......" << timeTable.size() - count << " more, "
            << timeTable.size() << " records in total\n";
      break;
    }
  }
}

// Singleton instance
std::unique_ptr<clProfiler> clProfiler::instance;

clProfiler *clProfiler::getInstance() {
  // Instance already exists
  if (instance.get()) return instance.get();

  // Create instance
  instance.reset(new clProfiler());
  return instance.get();
}

clProfiler::clProfiler() : strLen(16) {}

clProfiler::~clProfiler() {
  // Profiling info at the end of program execution
  if (getNumRecord()) getExecTime();
}

void clProfiler::getExecTime(std::string name) {
  if (name != "") {
    std::string sampleName = name;
    sampleName.resize(strLen, ' ');
    for (auto &meta : profilingData) {
      if (meta->getName() == sampleName) {
        std::cout << *meta;
      }
    }
  } else {
    double totalTime = 0.0f;
    std::cout << "Kernel Profiler info" << std::endl;
    for (auto &meta : profilingData) {
      std::cout << *meta;
      totalTime += meta->getTotalTime();
    }
    std::cout << "Kernel total time = " << std::setprecision(8)
              << totalTime * 1e-9 << " s/ " << totalTime << " ns" << std::endl;
  }
}

void clProfiler::addExecTime(std::string name, double st, double ed) {
  std::string sampleName = name;
  sampleName.resize(strLen, ' ');

  // Check if already in the list
  for (auto &elem : profilingData) {
    if (elem->getName() == sampleName) {
      elem->insert(st, ed);
      return;
    }
  }

  // Create if not in the list
  profilingData.emplace_back(new clProfilerMeta(sampleName));
  profilingData.back()->insert(st, ed);
}

double time_stamp_ms() {
  struct timeval t;
  if (gettimeofday(&t, 0) != 0) exit(-1);
  return t.tv_sec * 1e6 + t.tv_usec;
}

double time_stamp() {
  struct timeval t;
  if (gettimeofday(&t, 0) != 0) exit(-1);
  return t.tv_sec + t.tv_usec * 1e-6;
}

// Enqueue and profile a kernel
cl_int clProfileNDRangeKernel(cl_command_queue cmdQ, cl_kernel kernel,
                              cl_uint wd, const size_t *glbOs,
                              const size_t *glbSz, const size_t *locSz,
                              cl_uint numEvt, const cl_event *evtLst,
                              cl_event *evt) {
  cl_int err;
  cl_int enqueueErr;
  cl_event perfEvent;

  // Enqueue kernel
  enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0,
                                      NULL, &perfEvent);
  checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");
  clWaitForEvents(1, &perfEvent);

  // Get profiling information
  cl_ulong start = 0, end = 0;
  clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START,
                          sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end, NULL);

  // Get kernel name
  char kernelName[1024];
  err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char),
                        reinterpret_cast<void *>(kernelName), NULL);
  checkOpenCLErrors(err, "Failed to get kernel name");

  clProfiler *prof = clProfiler::getInstance();
  prof->addExecTime(kernelName, static_cast<double>(start),
                    static_cast<double>(end));

  return enqueueErr;
}

cl_int clTimeNDRangeKernel(cl_command_queue cmdQ, cl_kernel kernel, cl_uint wd,
                           const size_t *glbOs, const size_t *glbSz,
                           const size_t *locSz, cl_uint numEvt,
                           const cl_event *evtLst, cl_event *evt) {
  cl_int err;
  cl_int enqueueErr;

  clFinish(cmdQ);

  // Enqueue kernel
  double start = time_stamp_ms();
  enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0,
                                      NULL, NULL);
  clFinish(cmdQ);
  double end = time_stamp_ms();
  checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");

  // Get kernel name
  char kernelName[1024];
  err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char),
                        reinterpret_cast<void *>(kernelName), NULL);
  checkOpenCLErrors(err, "Failed to get kernel name");

  clProfiler *prof = clProfiler::getInstance();
  prof->addExecTime(kernelName, start, end);

  return enqueueErr;
}

}  // namespace clHelper
