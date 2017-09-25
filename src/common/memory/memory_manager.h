#ifndef SRC_COMMON_MEMORY_MANAGER_
#define SRC_COMMON_MEMORY_MANAGER_

template<typename T>
class Memory {
 protected:
  T *h_buf_;
  size_t count_;

 public:
  Memory(T *h_buf, size_t count)
      : h_buf_(h_buf), count_(count){};

  virtual ~Memory() {};

  /**
   * GetCount returns the number of elemnets in the 
   */
  virtual size_t GetCount() { return count_; };

  /**
   * GetHostPtr returns the pointer to the host memory
   */
  virtual T *GetHostPtr() { return h_buf_; };

  /**
   * GetDevicePtr returns the native reprentation of a device memory
   */
  virtual T *GetDevicePtr() = 0;

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

#endif  // SRC_COMMON_MEMORY_MANAGER_
