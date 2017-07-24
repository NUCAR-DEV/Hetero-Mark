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

#ifndef SRC_COMMON_CL_UTIL_CL_ERROR_H_
#define SRC_COMMON_CL_UTIL_CL_ERROR_H_

#include <CL/cl.h>
#include <iostream>
#include <string>

#define checkOpenCLSVMBuffer(buffer)                          \
  if (!(buffer))                                              \
    std::cout << "Location : " << __FILE__ << ":" << __LINE__ \
              << " invalid SVM buffer" << std::endl;          \
  exit(-1);

#define checkOpenCLErrors(actual, msg)                                      \
  if (checkVal(actual, CL_SUCCESS, msg)) {                                  \
    std::cout << "Location : " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(-1);                                                               \
  }

static inline void error(std::string errorMsg) {
  std::cout << "Error: " << errorMsg << std::endl;
}

// static const char* getOCLErrorCodeStr(std::string input)
// {
//     return "unknown error code";
// }

template <typename T>
static inline const char *getOCLErrorCodeStr(T input) {
  int errorCode = static_cast<int>(input);
  switch (errorCode) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
#endif
    default:
      return "UNKOWN ERROR CODE";
  }
}

template <typename T>
static int checkVal(T input, T reference, std::string message,
                    bool isAPIerror = true) {
  if (input == reference) {
    return 0;
  } else {
    if (isAPIerror) {
      std::cout << "Error: " << message << " Error code : ";
      std::cout << getOCLErrorCodeStr(input) << std::endl;
    } else {
      error(message);
    }

    return -1;
  }
}

#endif  // SRC_COMMON_CL_UTIL_CL_ERROR_H_
