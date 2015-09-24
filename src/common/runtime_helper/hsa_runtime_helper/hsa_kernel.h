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

#ifndef SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_KERNEL_H_
#define SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <hsa.h>

#include "src/common/runtime_helper/hsa_runtime_helper/aql_queue.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_agent.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_error_checker.h"

class HsaKernel {
 public:
  HsaKernel(hsa_executable_symbol_t symbol, HsaErrorChecker *error_checker);
  virtual ~HsaKernel() {}

  void SetKernelArgument(unsigned int index, unsigned int size_in_byte, 
      const void *value);
  void ExecuteKernel(HsaAgent *agent, AqlQueue *queue);
  void QueueKernel(HsaAgent *agent, AqlQueue *queue);
  void WaitKernel();

  void SetDimension(unsigned int dimension);
  void SetLocalSize(unsigned int dimension, unsigned int local_size);
  void SetGlobalSize(unsigned int dimension, unsigned int global_size);

 private:
  hsa_executable_symbol_t symbol_;
  HsaErrorChecker *error_checker_;
  hsa_status_t status_;

  uint64_t kernel_object_;
  uint32_t kernarg_segment_size_;
  uint32_t group_segment_size_;
  uint32_t private_segment_size_;

  // Maps from the index of an argument to its offset in kernarg memory
  std::vector<int> kernarg_index_to_offset_map_;
  std::unique_ptr<char> kernel_argument_value_;

  hsa_signal_t completion_signal_;
  // FIXME:Yifan This variable prevents kernel from running two instances 
  // at the same time. However, this should be allowed
  bool kernel_running_ = false;

  unsigned int dimension_ = 1;
  unsigned int global_size_[3] = {1, 1, 1};
  unsigned int local_size_[3] = {1, 1, 1};

  uint64_t GetKernelObjectFromSymbol();
  uint32_t GetKernargSegmentSizeFromSymbol();
  uint32_t GetPrivateSegmentSizeFromSymbol();
  uint32_t GetGroupSegmentSizeFromSymbol();

  void *AllocateKernargMemory(HsaAgent *agent, unsigned int size);
  static hsa_status_t GetKernargMemoryRegion(hsa_region_t region, void *data);
};

#endif  // SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_KERNEL_H_
