#ifndef HMM_FWDINITALPHAHSALAUNCHER_H
#define HMM_FWDINITALPHAHSALAUNCHER_H

#include "../common/HsaKernelLauncher.h"

class FwdInitAlphaHsaLauncher : public HsaKernelLauncher
{

	// Arguments
	struct args_t
	{
		uint64_t __global_offset_0;
		uint64_t __global_offset_1;
		uint64_t __global_offset_2;
		uint64_t __printf_buffer;
		uint64_t __vqueue_pointer;
		uint64_t __aqlwrap_pointer;
		uint32_t N;
		uint64_t b;
		uint64_t prior;
		uint64_t alpha;
		uint64_t beta;
	} args;

public:

	/**
	 * Constructor
	 */
	FwdInitAlphaHsaLauncher(HsaHelper *helper) : 
		HsaKernelLauncher(helper) {};

	/**
	 * Init
	 */
	void Init() override
	{
		name = "&__OpenCL_FWD_init_alpha_kernel";
		timer->BeginTimer();
		memset(&args, 0, sizeof(args_t));
		timer->EndTimer({"CPU", "memory"});
		HsaKernelLauncher::Init();
	}

	/**
	 * Launcher kernel
	 */
	void LaunchKernel() override
	{
		arguments = &args;
		HsaKernelLauncher::LaunchKernel();
	}

	/**
	 * Set argument
	 */
	void setArgument(int index, size_t size, void *value, 
			const char *option = NULL) override
	{
		switch (index)
		{
		case 0:
			args.N =  *(uint32_t *)value;
			break;
		case 1:
			memcpy(&args.b, value, size);
			//args.b =  *(uint64_t *)value;
			break;
		case 2:
			args.prior =  *(uint64_t *)value;
			break;
		case 3:
			args.alpha =  *(uint64_t *)value;
			break;
		case 4:
			args.beta =  *(uint64_t *)value;
			break;
		default:
			printf("Invalid argument index %d.\n", index);
			exit(1);
		}
	}
};

#endif
