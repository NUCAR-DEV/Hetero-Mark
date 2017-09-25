#ifndef SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_
#define SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_

#include <hcc/hc.hpp>

#include "src/common/memory/memory_manager.h"

#if COMPILE_HCC

template <typename T, int N>
class ArrayMemory : public Memory<T> {
 protected:
  hc::array<T, N> d_array_;

 public:
  ArrayMemory(T *h_buf, size_t count)
      : Memory<T>(h_buf, count), d_array_(count) {
  };

  T *GetDevicePtr() override {
    return d_array_.accelerator_pointer();
  }

  hc::array<T, N> GetNative() { return d_array_; };

  void HostToDevice() override { 
    hc::copy(this->h_buf_, this->h_buf_ + this->count_, d_array_);
  }

  void DeviceToHost() override { 
    hc::copy(d_array_, this->h_buf_);
  }

  void Free() override {
    // The GPU memory will be freed when the array is destructed.
  }
};

class ArrayMemoryManager{
 public:
  template <typename T, int N>
  ArrayMemory<T, N> *Shadow(T *buf, size_t count) {
    return new ArrayMemory<T, N>(buf, count);
  }
};

#endif  // COMPILE_HCC

#endif  // SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_
