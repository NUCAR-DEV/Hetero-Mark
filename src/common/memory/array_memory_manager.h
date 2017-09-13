#include <hcc/hc.hpp>

#include "src/common/memory/memory_manager.h"

#if COMPILE_HCC

class ArrayMemory : public Memory {
 protected:
  hc::array<uint8_t, 1> d_array_;

 public:
  ArrayMemory(void *h_buf, size_t byte_size)
      : Memory(h_buf, byte_size), d_array_(byte_size) {
  };

  void *GetDevicePtr() override {
    return static_cast<void *>(d_array_.accelerator_pointer());
  }

  void *GetDevicePtrOnDevice() [[hc]] {
    return static_cast<void *>(d_array_.accelerator_pointer());
  }

  void HostToDevice() override { 
    hc::copy((uint8_t *)h_buf_, (uint8_t *)h_buf_ + byte_size_, d_array_);
  }

  void DeviceToHost() override { 
    hc::copy(d_array_, (uint8_t *)h_buf_);
  }

  void Free() override {
    // The GPU memory will be freed when the array_view is destructed.
  }
};

class ArrayMemoryManager : public MemoryManager {
 public:
  ArrayMemory *Shadow(void *buf, size_t byte_size) {
    return new ArrayMemory(buf, byte_size);
  }
};

#endif
