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
#ifndef SRC_COMMON_CL_UTIL_CL_RUNTIME_H_
#define SRC_COMMON_CL_UTIL_CL_RUNTIME_H_

#include <CL/cl.h>
#include <memory>
#include <string>
#include <vector>

#include "src/common/cl_util/cl_error.h"

namespace clHelper {

enum clSVMLevel { SVM_COARSE, SVM_FINE, SVM_SYSTEM, SVM_ATOMIC };

// OpenCL runtime contains objects don't change much during execution
// These objects are automatically freed in the destructor
class clRuntime {
 private:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;

  std::vector<cl_command_queue> cmdQueueRepo;

  // Instance of the singleton
  static std::unique_ptr<clRuntime> instance;

  // Private constructor for singleton
  clRuntime();

  int displayPlatformInfo(cl_platform_id plt_id, cl_platform_info plt_info);

  int displayContextInfo(cl_context ctx, cl_context_info ctx_info);

  int displayDeviceInfo(cl_context ctx, cl_context_info device_info);

  void requireCL20();

 public:
  // Destructor
  ~clRuntime();

  // Get singleton
  static clRuntime *getInstance();

  /// Getters
  cl_platform_id getPlatformID() const { return platform; }

  cl_device_id getDevice() const { return device; }

  cl_context getContext() const { return context; }

  // Get a command queue by index, create it if doesn't exist
  cl_command_queue getCmdQueue(unsigned index,
                               cl_command_queue_properties properties = 0);

  // Get number of compute units of current device
  cl_uint getNumComputeUnit() const;

  // Device SVM support
  bool isSVMavail(enum clSVMLevel level);

  // Print information of the platform
  int displayPlatformInfo();

  int displayDeviceInfo();

  int displayContextInfo();

  int displayAllInfo();
};

}  // namespace clHelper

#endif  // SRC_COMMON_CL_UTIL_CL_RUNTIME_H_
