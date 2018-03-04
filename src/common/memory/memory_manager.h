#ifndef SRC_COMMON_MEMORY_MANAGER_
#define SRC_COMMON_MEMORY_MANAGER_

class Memory {
 protected:
  void *h_buf_;
  size_t byte_size_;

 public:
  Memory(void *h_buf, size_t byte_size)
      : h_buf_(h_buf), byte_size_(byte_size){};

  virtual ~Memory() {};

  /**
   * GetByteSize returns the number of bytes occupied by the memory
   */
  virtual size_t GetByteSize() { return byte_size_; };

  /**
   * GetHostPtr returns the pointer to the host memory
   */
  virtual void *GetHostPtr() { return h_buf_; };

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

class MemoryManager{
 public:
   virtual ~MemoryManager() {}
   virtual std::unique_ptr<Memory> Shadow(void *buf, size_t byte_size) = 0;
};


#endif  // SRC_COMMON_MEMORY_MANAGER_
