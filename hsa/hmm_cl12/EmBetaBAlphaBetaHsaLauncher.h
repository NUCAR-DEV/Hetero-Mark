#ifndef HMM_EMBETABALPHAHSALAUNCHER_H
#define HMM_EMBETABALPHAHSALAUNCHER_H

#include "../common/HsaKernelLauncher.h"

class EmBetaBAlphaBetaHsaLauncher : public HsaKernelLauncher
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
		uint32_t current;
		uint32_t previous;
		void *beta;
		void *b;
		void *alpha;
		void *betaB;
		void *alpha_beta;
	} args;

public:

	/**
	 * Constructor
	 */
	EmBetaBAlphaBetaHsaLauncher(HsaHelper *helper) : 
		HsaKernelLauncher(helper) {};

	/**
	 * Init
	 */
	void Init() override
	{
		name = "&__OpenCL_EM_betaB_alphabeta_kernel";
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
		for(int i = 0; i < sizeof(args) / 4; i++)
		{
			printf("0x%08x\n", *((uint32_t *)(&args) + i));
		}
		printf("Launching kernel %s\n", name.c_str());
		printf("Size of args: %ld\n", sizeof(args));
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
			printf("&args.N: %p\n", &args.N);
			break;
		case 1:
			memcpy(&args.current, value, size);
			printf("&args.current: %p\n", &args.current);
			break;
		case 2:
			memcpy(&args.previous, value, size);
			printf("&args.previous: %p\n", &args.previous);
			break;
		case 3:
			memcpy(&args.beta, value, size);
			printf("&args.beta: %p\n", &args.beta);
			break;
		case 4:
			memcpy(&args.b, value, size);
			printf("&args.b: %p\n", &args.b);
			break;
		case 5:
			memcpy(&args.alpha, value, size);
			printf("&args.alpha: %p\n", &args.alpha);
			break;
		case 6:
			memcpy(&args.betaB, value, size);
			printf("&args.betaB: %p\n", &args.betaB);
			break;
		case 7:
			printf("Alpha_beta: %p\n", *(void **)value);
			printf("&args.alpha_beta: %p\n", &args.alpha_beta);
			memcpy(&args.alpha_beta, value, size);
			break;
		default:
			printf("Invalid argument index %d.\n", index);
			exit(1);
		}

		for(int i = 0; i < sizeof(args) / 4; i++)
		{
			printf("0x%08x\n", *((uint32_t *)(&args) + i));
		}
		printf("\n");

		timer->EndTimer({"CPU", "memory"});
	}
};

#endif
