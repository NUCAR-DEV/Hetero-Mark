#ifndef SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_
#define SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_

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
  };

  T *GetDevicePtr() override {
    return d_buf_.accelerator_pointer();
  }

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
#endif  // SRC_COMMON_MEMORY_ARRAY_VIEW_MEMORY_MANAGER_
