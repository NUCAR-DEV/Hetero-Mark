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

#ifndef SRC_COMMON_MEMORY_MEMORY_MANAGER_H_
#define SRC_COMMON_MEMORY_MEMORY_MANAGER_H_

#include <cstdlib>
#include <memory>

class Memory {
 protected:
  void *h_buf_;
  size_t byte_size_;

 public:
  Memory(void *h_buf, size_t byte_size)
      : h_buf_(h_buf), byte_size_(byte_size) {}

  virtual ~Memory() {}

  /**
   * GetByteSize returns the number of bytes occupied by the memory
   */
  virtual size_t GetByteSize() { return byte_size_; }

  /**
   * GetHostPtr returns the pointer to the host memory
   */
  virtual void *GetHostPtr() { return h_buf_; }

  /**
   * GetDevicePtr returns the native reprentation of a device memory
   */
  virtual void *GetDevicePtr() = 0;

  /**
   * HostToDevice copies data from the host to the device
   */
  virtual void HostToDevice() = 0;

  /**
   * DeviceToHost copies data from the device to the host
   */
  virtual void DeviceToHost() = 0;

  /**
   * Free releases the memory on the GPU
   */
  virtual void Free() = 0;
};

class MemoryManager {
 public:
  virtual ~MemoryManager() {}
  virtual std::unique_ptr<Memory> Shadow(void *buf, size_t byte_size) = 0;
};

#endif  // SRC_COMMON_MEMORY_MEMORY_MANAGER_H_
