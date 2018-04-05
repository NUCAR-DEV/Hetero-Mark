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

#ifndef SRC_COMMON_MEMORY_HSA_SVM_MEMORY_MANAGER_H_
#define SRC_COMMON_MEMORY_HSA_SVM_MEMORY_MANAGER_H_

#if COMPILE_HSA

#include <hsa/hsa.h>
#include <memory>

#include "src/common/memory/memory_manager.h"

class HsaMemory : public Memory {
 public:
  HsaMemory(void *h_buf, size_t byte_size) : Memory(h_buf, byte_size) {
    hsa_status_t err;
    err = hsa_memory_register(h_buf, byte_size);
    if (err != HSA_STATUS_SUCCESS) {
      std::cerr << "Failed to register HSA memory, error: " << err << "\n";
      exit(-1);
    }
  }

  void *GetDevicePtr() override { return h_buf_; }

  void HostToDevice() override {
    // No need to do anything
  }

  void DeviceToHost() override {
    // No need to do anything
  }

  void Free() override {
    hsa_status_t err;
    err = hsa_memory_deregister(h_buf_, byte_size_);
    if (err != HSA_STATUS_SUCCESS) {
      std::cerr << "Failed to register HSA memory, error: " << err << "\n";
      exit(-1);
    }
  }
};

class HsaMemoryManager : public MemoryManager {
 public:
  std::unique_ptr<Memory> Shadow(void *buf, size_t byte_size) {
    return std::unique_ptr<Memory>(new HsaMemory(buf, byte_size));
  }
};

#endif  // COMPILE_HSA

#endif  // SRC_COMMON_MEMORY_HSA_SVM_MEMORY_MANAGER_H_
