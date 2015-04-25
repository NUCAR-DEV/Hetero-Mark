#ifndef HMM_EMEXPECTAHSALAUNCHER_H
#define HMM_EMEXPECTAHSALAUNCHER_H

#include "../common/HsaKernelLauncher.h"

class EmExpectAHsaLauncher : public HsaKernelLauncher
{

	// Arguments
	struct __attribute__ ((aligned(16))) args_t
	{
		uint64_t __global_offset_0;
		uint64_t __global_offset_1;
		uint64_t __global_offset_2;
		uint64_t __printf_buffer;
		uint64_t __vqueue_pointer;
		uint64_t __aqlwrap_pointer;
		uint32_t N;
		void *sm;
		void *xi_sum;
		void *expect_A;
	} args;

public:

	/**
	 * Constructor
	 */
	EmExpectALauncher(HsaHelper *helper) : 
		HsaKernelLauncher(helper) {};

	/**
	 * Init
	 */
	void Init() override
	{
		name = "&__OpenCL_EM_expectA_kernel";
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
		//printf("Launching fwd init alpha kernel\n");
		HsaKernelLauncher::LaunchKernel();
	}

	/**
	 * Set argument
	 */
	void setArgument(int index, size_t size, void *value, 
			const char *option = NULL) override
	{
		timer->BeginTimer();
		switch (index)
		{
		case 0:
			memcpy(&args.N, value, size);
			break;
		case 1:
			setGroupSegmentSize(size);
			break;
		case 2:
			memcpy(&args.xi_sum, value, size);
			break;
		case 3:
			memcpy(&args.expectA, value, size);
			break;
		default:
			printf("Invalid argument index %d.\n", index);
			exit(1);
		}
		timer->EndTimer({"CPU", "memory"});
	}
};

#endif
