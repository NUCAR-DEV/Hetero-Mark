#ifndef HMM_FWDINITALPHAHSALAUNCHER_H
#define HMM_FWDINITALPHAHSALAUNCHER_H

#include "../common/HsaKernelLauncher.h"

class FwdInitAlphaHsaLauncher : public HsaKernelLauncher
{
public:

	/**
	 * Launcher kernel
	 */
	void LaunchKernel() override
	{
	}

	/**
	 * Set argument
	 */
	void setArgument(int index, size_t size, void *value, 
			const char *option = NULL) override
	{
	}
};

#endif
