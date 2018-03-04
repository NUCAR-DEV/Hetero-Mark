#ifndef SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_
#define SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_

#if COMPILE_HSA

#include <hsa/hsa.h>

#include "src/common/memory/memory_manager.h"

class HsaMemory : public Memory {
 protected:
 public:
  HsaMemory(void *h_buf, size_t byte_size) : Memory(h_buf, byte_size) {
    hsa_status_t err;
    err = hsa_memory_register(h_buf, byte_size);
    if (err != HSA_STATUS_SUCCESS) {
      std::cerr << "Failed to register HSA memory, error: " << err << "\n";
      exit(-1);
    }
  };

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

#endif  // SRC_COMMON_MEMORY_ARRAY_MEMORY_MANAGER_
