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

#include "src/common/cl_util/cl_runtime.h"

#include <memory>

namespace clHelper {

// Singleton instance
std::unique_ptr<clRuntime> clRuntime::instance;

clRuntime *clRuntime::getInstance() {
  // Instance already exists
  if (instance.get()) return instance.get();

  // Create instance
  instance.reset(new clRuntime());
  return instance.get();
}

clRuntime::clRuntime() {
  cl_int err = 0;

  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  checkOpenCLErrors(err, "Failed at clGetPlatformIDs");

  // Get ID for the device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  checkOpenCLErrors(err, "Failed at clGetDeviceIDs");

  // Create a context
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  checkOpenCLErrors(err, "Failed at clCreateContext");
}

clRuntime::~clRuntime() {
  cl_int err = 0;

  for (auto &cmdQueue : cmdQueueRepo) {
    err = clReleaseCommandQueue(cmdQueue);
    checkOpenCLErrors(err, "Failed at clReleaseCommandQueue");
  }

  if (context) {
    err = clReleaseContext(context);
    checkOpenCLErrors(err, "Failed at clReleaseContext");
  }

  if (device) {
    err = clReleaseDevice(device);
    checkOpenCLErrors(err, "Failed at clReleaseDevice");
  }
}

void clRuntime::requireCL20() {
  cl_int err;
  char decVer[256];
  char langVer[256];

  err =
      clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(decVer), decVer, NULL);
  checkOpenCLErrors(err, "Failed to clGetDeviceInfo: CL_DEVICE_VERSION")

      err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(langVer),
                            langVer, NULL);
  checkOpenCLErrors(err,
                    "Failed to clGetDeviceInfo: CL_DEVICE_OPENCL_C_VERSION")

      std::string dev_version(decVer);
  std::string lang_version(langVer);
  dev_version = dev_version.substr(std::string("OpenCL ").length());
  dev_version = dev_version.substr(0, dev_version.find('.'));

  // The format of the version string is defined in the spec as
  // "OpenCL C <major>.<minor> <vendor-specific>"
  lang_version = lang_version.substr(std::string("OpenCL C ").length());
  lang_version = lang_version.substr(0, lang_version.find('.'));

  if (!(stoi(dev_version) >= 2 && stoi(lang_version) >= 2)) {
    printf(
        "Device does not support OpenCL 2.0!\n \tCL_DEVICE_VERSION: %s, "
        "CL_DEVICE_OPENCL_C_VERSION, %s\n",
        decVer, langVer);
    exit(-1);
  }
}

int clRuntime::displayPlatformInfo(cl_platform_id plt_id,
                                   cl_platform_info plt_info) {
  cl_int err;
  char platformInfo[1024];
  err = clGetPlatformInfo(plt_id, plt_info, sizeof(platformInfo), platformInfo,
                          NULL);
  checkOpenCLErrors(err, "clGetPlatformInfo failed");
  std::cout << "\t" << platformInfo << std::endl;

  return 0;
}

int clRuntime::displayContextInfo(cl_context ctx, cl_context_info ctx_info) {
  return 0;
}

int clRuntime::displayPlatformInfo() {
  std::cout << "Platform info:" << std::endl;
  displayPlatformInfo(platform, CL_PLATFORM_VENDOR);
  displayPlatformInfo(platform, CL_PLATFORM_VERSION);
  displayPlatformInfo(platform, CL_PLATFORM_PROFILE);
  displayPlatformInfo(platform, CL_PLATFORM_NAME);
  displayPlatformInfo(platform, CL_PLATFORM_EXTENSIONS);

  return 0;
}

int clRuntime::displayDeviceInfo() {
  cl_int err;

  // Get number of devices available
  cl_uint deviceCount = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
  checkOpenCLErrors(err, "Failed at clGetDeviceIDs");

  // Get device ids
  cl_device_id *deviceIds = reinterpret_cast<cl_device_id *>(
      malloc(sizeof(cl_device_id) * deviceCount));
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds,
                       NULL);
  checkOpenCLErrors(err, "Failed at clGetDeviceIDs");

  // Print device index and device names
  std::cout << "Devices info:" << std::endl;
  for (cl_uint i = 0; i < deviceCount; ++i) {
    char deviceName[1024];
    cl_ulong maxmembytes;

    err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, sizeof(deviceName),
                          deviceName, NULL);
    checkOpenCLErrors(err, "Failed at clGetDeviceInfo");

    err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(cl_ulong), &maxmembytes, NULL);

    std::cout << "(*)\tDevice " << i << " = " << deviceName
              << "\n\tDevice ID = " << deviceIds[i] << "\n\tMax Memory = "
              << static_cast<float>(maxmembytes) / 1048576 << "MB" << std::endl;
  }

  free(deviceIds);

  return 0;
}

int clRuntime::displayAllInfo() {
  displayPlatformInfo();
  displayDeviceInfo();

  return 0;
}

cl_command_queue clRuntime::getCmdQueue(
    unsigned index, cl_command_queue_properties properties) {
  cl_int err;

  if (index < cmdQueueRepo.size()) {
    return cmdQueueRepo[index];
  } else {
#ifdef CL_VERSION_2_0
    std::vector<cl_queue_properties> queue_properties;

    if (properties) {
      queue_properties.push_back(CL_QUEUE_PROPERTIES);
      queue_properties.push_back(cl_queue_properties(properties));
      queue_properties.push_back(cl_queue_properties(0));
    }

    const cl_queue_properties *queue_properties_ptr =
        queue_properties.empty() ? 0 : &queue_properties[0];

    cl_command_queue cmdQ = clCreateCommandQueueWithProperties(
        context, device, queue_properties_ptr, &err);
    checkOpenCLErrors(err, "Failed at clCreateCommandQueueWithProperties");
#else
    cl_command_queue cmdQ =
        clCreateCommandQueue(context, device, properties, &err);
    checkOpenCLErrors(err, "Failed at clCreateCommandQueue");
#endif

    cmdQueueRepo.push_back(cmdQ);
    return cmdQ;
  }
}

bool clRuntime::isSVMavail(enum clSVMLevel level) {
#ifdef CL_VERSION_2_0
  cl_device_svm_capabilities caps;

  cl_int err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                               sizeof(cl_device_svm_capabilities), &caps, 0);
  checkOpenCLErrors(err,
                    "Failed at clGetDeviceInfo: CL_DEVICE_SVM_CAPABILITIES");

  switch (level) {
    case SVM_COARSE:
      return caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? true : false;

    case SVM_FINE:
      return caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? true : false;

    case SVM_SYSTEM:
      return caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? true : false;

    case SVM_ATOMIC:
      return caps & CL_DEVICE_SVM_ATOMICS ? true : false;

    default:
      return false;
  }
#else
  return false;
#endif
}

cl_uint clRuntime::getNumComputeUnit() const {
  cl_int err;
  cl_uint numComputeUnit;

  err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                        &numComputeUnit, NULL);
  checkOpenCLErrors(err,
                    "Failed at clGetDeviceInfo: CL_DEVICE_MAX_COMPUTE_UNITS");

  return numComputeUnit;
}

}  // namespace clHelper
