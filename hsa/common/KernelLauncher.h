#ifndef COMMON_KERNELLAUNCHER_H
#define COMMON_KERNELLAUNCHER_H

#include <cstdint>

/**
 * A kernel launcher is an abstract class that helps launch a kernel.
 * Each kernel should have its own implementation
 */
class KernelLauncher
{
protected:

	// Global size
	uint32_t global_size[3];

	// Group size
	uint32_t group_size[3];

public:

	/**
	 * Init Launcher
	 */
	virtual void Init() = 0;

	/**
	 * Launch the kernel
	 */
	virtual void LaunchKernel() = 0;

	/**
	 * Set group size
	 */
	void setGroupSize(uint32_t x, uint32_t y, uint32_t z)
	{
		group_size[0] = x;
		group_size[1] = y;
		group_size[2] = z;
	}

	/**
	 * Set global size
	 */
	void setGlobalSize(uint32_t x, uint32_t y, uint32_t z)
	{
		global_size[0] = x;
		global_size[1] = y;
		global_size[2] = z;
	}

	/**
	 * Set argument 
	 */
	virtual void setArgument(int index, size_t size, void *value,
			const char *option = NULL) = 0;


};

#endif
