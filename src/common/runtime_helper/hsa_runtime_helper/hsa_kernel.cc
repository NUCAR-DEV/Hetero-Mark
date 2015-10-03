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

#include <cstring>

#include "src/common/runtime_helper/hsa_runtime_helper/hsa_kernel.h"

HsaKernel::HsaKernel(hsa_executable_symbol_t symbol, 
  HsaErrorChecker *error_checker) :
  symbol_(symbol),
  error_checker_(error_checker) {

  // Variables from the symbol
  kernel_object_ = GetKernelObjectFromSymbol();
  kernarg_segment_size_ = GetKernargSegmentSizeFromSymbol();
  private_segment_size_ = GetPrivateSegmentSizeFromSymbol();
  group_segment_size_ = GetGroupSegmentSizeFromSymbol();

  // Kernel argument space
  kernel_argument_value_.reset(new char[kernarg_segment_size_]());
  kernarg_index_to_offset_map_.push_back(48);
}


void HsaKernel::SetKernelArgument(unsigned int index, 
    unsigned int size_in_byte, const void *value) {
  /*
  std::cerr << "Setting argument " << index << ".\n";
  std::cerr << "Known offsets: \n";
  for (unsigned int i = 0; i < kernarg_index_to_offset_map_.size(); i++) {
    std::cerr << "Index: " << i << " , offset: " << 
      kernarg_index_to_offset_map_[i] << "\n";
  }
  */

  // Prevent undefined result
  if (index > kernarg_index_to_offset_map_.size()) {
    std::cerr << "Our runtime helper need to set kernel argument in order.\n";
    exit(1);
  }

  // Get the offset
  int offset = kernarg_index_to_offset_map_[index - 1];
  memcpy(kernel_argument_value_.get() + offset, value, size_in_byte);

  // Set next arguments offset
  if (offset + size_in_byte < kernarg_segment_size_ && 
      index == kernarg_index_to_offset_map_.size()) {
    kernarg_index_to_offset_map_.push_back(offset + size_in_byte);
  }
}

void HsaKernel::ExecuteKernel(HsaAgent *agent, AqlQueue *queue) {
  QueueKernel(agent, queue);
  WaitKernel();
}

void HsaKernel::QueueKernel(HsaAgent *agent, AqlQueue *queue) {
  // Disallow kernel running multiple instances
  if (kernel_running_) {
    std::cerr << "Kernel is running.\n";
    exit(1);
  }

  // Create the signal
  status_ = hsa_signal_create(1, 0, NULL, &completion_signal_);
  error_checker_->SucceedOrDie("Creating completion signal", status_);

  // Allocate memory 
  void *kernarg_address = AllocateKernargMemory(agent, kernarg_segment_size_);

  // Copy kernel arguments into allocated memory
  memcpy(kernarg_address, kernel_argument_value_.get(), kernarg_segment_size_);

  // Get native queue
  hsa_queue_t *native_queue = (hsa_queue_t *)queue->GetNative();

  uint64_t index = hsa_queue_load_write_index_relaxed(native_queue);
  const uint32_t queue_mask = native_queue->size - 1;
  hsa_kernel_dispatch_packet_t *dispatch_packet = 
    &(((hsa_kernel_dispatch_packet_t *)(native_queue->base_address))
        [index&queue_mask]);
  dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << 
    HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << 
    HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  dispatch_packet->setup  |= dimension_ << 
    HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch_packet->workgroup_size_x = (uint16_t)local_size_[0];
  dispatch_packet->workgroup_size_y = (uint16_t)local_size_[1];
  dispatch_packet->workgroup_size_z = (uint16_t)local_size_[2];
  dispatch_packet->grid_size_x = (uint32_t)global_size_[0];
  dispatch_packet->grid_size_y = (uint32_t)global_size_[1];
  dispatch_packet->grid_size_z = (uint32_t)global_size_[2];
  dispatch_packet->completion_signal = completion_signal_;
  dispatch_packet->kernel_object = kernel_object_;
  dispatch_packet->kernarg_address = (void*) kernarg_address;
  dispatch_packet->private_segment_size = private_segment_size_;
  dispatch_packet->group_segment_size = group_segment_size_;
  __atomic_store_n((uint8_t*)(&dispatch_packet->header), 
      (uint8_t)HSA_PACKET_TYPE_KERNEL_DISPATCH, __ATOMIC_RELEASE);

  hsa_queue_store_write_index_relaxed(native_queue, index+1);
  hsa_signal_store_relaxed(native_queue->doorbell_signal, index);
  error_checker_->SucceedOrDie("Dispatching the kernel", status_);
};

void HsaKernel::WaitKernel() {
  hsa_signal_wait_acquire(completion_signal_, HSA_SIGNAL_CONDITION_LT, 1, 
      UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
};

uint64_t HsaKernel::GetKernelObjectFromSymbol() {
  uint64_t kernel_object;
  status_ = hsa_executable_symbol_get_info(symbol_, 
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  error_checker_->SucceedOrDie("Extracting the symbol from the executable", 
      status_);
  return kernel_object;
}

uint32_t HsaKernel::GetKernargSegmentSizeFromSymbol() {
  uint32_t kernarg_segment_size;
  status_ = hsa_executable_symbol_get_info(symbol_, 
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, 
      &kernarg_segment_size);
  error_checker_->SucceedOrDie("Extracting the kernarg segment size from the "
      "executable", status_);
  return kernarg_segment_size;
}

uint32_t HsaKernel::GetPrivateSegmentSizeFromSymbol() {
  uint32_t private_segment_size;
  status_ = hsa_executable_symbol_get_info(symbol_, 
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, 
      &private_segment_size);
  error_checker_->SucceedOrDie("Extracting the private segment size from the "
      "executable", status_);
  return private_segment_size;
}

uint32_t HsaKernel::GetGroupSegmentSizeFromSymbol() {
  uint32_t group_segment_size;
  status_ = hsa_executable_symbol_get_info(symbol_, 
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, 
      &group_segment_size);
  error_checker_->SucceedOrDie("Extracting the group segment size from the "
      "executable", status_);
  return group_segment_size;
}

void *HsaKernel::AllocateKernargMemory(HsaAgent *agent, unsigned int size) {
  // Find memory region that support kernarg
  hsa_region_t kernarg_region;
  kernarg_region.handle=(uint64_t)-1;
  hsa_agent_iterate_regions(*(hsa_agent_t *)agent->GetNative(), 
      GetKernargMemoryRegion, &kernarg_region);
  status_ = (kernarg_region.handle == (uint64_t)-1) ? 
    HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
  error_checker_->SucceedOrDie("Finding a kernarg memory region", status_);

  // Allocate
  void* kernarg_address = NULL;
  status_ = hsa_memory_allocate(kernarg_region, size, 
      &kernarg_address);
  error_checker_->SucceedOrDie("Allocating kernel argument memory buffer", 
      status_);

  return kernarg_address;
}

hsa_status_t HsaKernel::GetKernargMemoryRegion(
    hsa_region_t region, void *data) {
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (HSA_REGION_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t* ret = (hsa_region_t*) data;
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

void HsaKernel::SetDimension(unsigned int dimension) {
  if (dimension > 3 || dimension < 1) {
    std::cerr << "Dimension can only be 1, 2, or 3.\n";
    exit(1);
  } 
  dimension_ = dimension;
}

void HsaKernel::SetLocalSize(unsigned int dimension, unsigned int local_size) {
  if (dimension < 1 || dimension > dimension_) {
    std::cerr << "Cannot set local size on dimension " << dimension << ".\n";
    exit(1);
  }
  local_size_[dimension - 1] = local_size;
}

void HsaKernel::SetGlobalSize(unsigned int dimension, 
    unsigned int global_size) {
  if (dimension < 1 || dimension > dimension_) {
    std::cerr << "Cannot set global size on dimension " << dimension << ".\n";
    exit(1);
  }
  global_size_[dimension - 1] = global_size;
}
