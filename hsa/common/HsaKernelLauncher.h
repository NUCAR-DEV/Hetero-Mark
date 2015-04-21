#ifndef COMMON_HSAKERNELLAUNCHER_H
#define COMMON_HSAKERNELLAUNCHER_H

#include <string>
#include "hsa.h"
#include "hsa_ext_finalize.h"

#include "TimeKeeper.h"

#include "KernelLauncher.h"

class HsaHelper;

/**
 * An HSA kernel launcher help launch hsa kernel
 */
class HsaKernelLauncher : public KernelLauncher
{
protected:
	// Timer
	TimeKeeper *timer;

	// Hsa Helper
	HsaHelper *helper;
	
	// Name of the kernel to launch
	std::string name;

	// Kernel object
	uint64_t kernel_object;

	// Kernel argument address
	void *kernarg_address;

	// Arguments
	void *arguments;

	// Kernal argument size
	uint32_t kernarg_segment_size;

	// Private segment size
	uint32_t private_segment_size;

	// Group segment size
	uint32_t group_segment_size;

	// Callback function to get a memory that supports kernel argument 
	static hsa_status_t FindKernargMemoryRegion(hsa_region_t region, 
			void *data);

	// Create argument buffer and copy the argument
	void PrepareArgument();

public:

	/**
	 * Constructor
	 */
	HsaKernelLauncher() 
	{
		timer = TimeKeeper::getInstance();
	}

	/**
	 * Init the hsa kernel launcher
	 */
	void Init() override;

	/**
	 * Launch kernel
	 */
	void LaunchKernel() override;

	/**
	 * Set private segment size
	 */
	void setPrivateSegmentSize(uint32_t size) 
	{ 
		private_segment_size = size; 
	}

	/**
	 * Set group segment size
	 */
	void setGroupSegmentSize(uint32_t size) { group_segment_size = size; }

	/**
	 * Set name
	 */ 
	void setName(const char *name) { this->name = std::string(name); }

	/**
	 * Set helper
	 */
	void setHelper(HsaHelper *helper) { this->helper = helper; }

	/**
	 * Set arguments
	 */
	void setArguments(void *args) { this->arguments = args; }

	/**
	 * Set argument 
	 */
	virtual void setArgument(int index, size_t size, void *value,
			const char *option = NULL);
};

#endif
