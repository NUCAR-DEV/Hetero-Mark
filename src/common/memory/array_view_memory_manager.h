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

#ifndef SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_H_
#define SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_H_

#include <hcc/hc.hpp>

#include "src/common/memory/memory_manager.h"

#if COMPILE_HCC

template <typename T, int N>
class ArrayViewMemory : public Memory<T> {
 public:
  hc::array_view<T, N> d_buf_;

  ArrayViewMemory(T *h_buf, size_t count)
      : Memory<T>(h_buf, count), d_buf_(count, h_buf) {
    d_buf_.synchronize_to(hc::accelerator().get_default_view());
  }

  T *GetDevicePtr() override { return d_buf_.accelerator_pointer(); }

  void HostToDevice() override { d_buf_.refresh(); }

  void DeviceToHost() override {
    d_buf_.synchronize_to(hc::accelerator(L"cpu").get_default_view());
    d_buf_.synchronize();
  }

  void Free() override {
    // The GPU memory will be freed when the array_view is destructed.
  }
};

class ArrayViewMemoryManager {
 public:
  template <typename T, int N>
  ArrayViewMemory<T, N> *Shadow(T *buf, size_t count) {
    return new ArrayViewMemory<T, N>(buf, count);
  }
};

#endif  // COMPILE_HCC
#endif  // SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_H_
