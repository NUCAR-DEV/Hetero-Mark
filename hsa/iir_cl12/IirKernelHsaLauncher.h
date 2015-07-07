#ifndef IIR_CL20_IIRKERNELHSALAUNCHER_H
#define IIR_CL20_IIRKERNELHSALAUNCHER_H

#include "../common/HsaKernelLauncher.h"

class IirKernelHsaLauncher : public HsaKernelLauncher
{

	// Arguments
	struct __attribute__ ((aligned(16))) args_t 
	{
		uint64_t global_offset_0;
		uint64_t global_offset_1;
		uint64_t global_offset_2;
		uint64_t printf_buffer;
		uint64_t vqueue_pointer;
		uint64_t aqlwrap_pointer;
		uint32_t len;
		float c;
		void *nsec;
		uint32_t pad1;
		void *dsec;
		uint32_t pad2;
		void *sm;
		uint32_t pad3;
		void *x;
		uint32_t pad4;
		void *y;
		uint32_t pad5;
	} args;

public:

	/**
	 * Constructor
	 */
	IirKernelHsaLauncher(HsaHelper *helper) : 
		HsaKernelLauncher(helper) {};

	/**
	 * Init
	 */
	void Init() override
	{
		name = "&__OpenCL_ParIIR_kernel";
		timer->BeginTimer();
		memset(&args, 0, sizeof(args));
		timer->EndTimer({"CPU", "memory"});
		HsaKernelLauncher::Init();
	}

	/**
	 * Launcher kernel
	 */
	void LaunchKernel() override
	{
		arguments = &args;
		setGroupSegmentSize(1024 * sizeof(float));
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
			memcpy(&args.len, value, size);
			break;
		case 1:
			memcpy(&args.c, value, size);
			break;
		case 2:
			memcpy(&args.nsec, value, size);
			break;
		case 3:
			memcpy(&args.dsec, value, size);
			break;
		case 4:
			args.sm =  (void *)(4 * sizeof(float));
			break;
		case 5:
			memcpy(&args.x, value, size);
			break;
		case 6:
			memcpy(&args.y, value, size);
			break;
		default:
			printf("Invalid argument index %d.\n", index);
			exit(1);
		}
	}
};

#endif
