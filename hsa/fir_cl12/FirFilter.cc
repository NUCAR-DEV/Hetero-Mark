#include <cstring>

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "FirFilter.h"

FirFilter::FirFilter() :
	kernel_launcher()
{
}


void FirFilter::Init() 
{
	// Init helper
	helper->Init();

	// Load kernels
	helper->LoadProgram("kernel.brig");

	// Setup kernel launcher
	kernel_launcher.setHelper(helper);
	kernel_launcher.setName("&__OpenCL_FIR_kernel");
	kernel_launcher.Init();

	// Init arguments
	InitParam();
}


void FirFilter::InitParam()
{
	// Reset
	memset(&args, 0, sizeof(args));

	// Input and output argument
	in = (float *)malloc(sizeof(float) * len);
	helper->RegisterMemory(in, sizeof(float) * len);
	out = (float *)malloc(sizeof(float) * len);
	helper->RegisterMemory(out, sizeof(float) * len);
	for (int i = 0; i < len; i++)
	{
		in[i] = (float)i;
	}

	// Filter parameters
	numTap = 4;
	coeff = (float *)malloc(sizeof(float) * numTap);
	helper->RegisterMemory(coeff, sizeof(float) * numTap);
	for (int i = 0; i < numTap; i++)
	{
		coeff[i] = i;
	}
	

	// Write args	
	args.output = out;
	args.input = in;
	args.coeff = coeff;
	args.numTap = numTap;

	// Set to kernel launcher
	kernel_launcher.setArguments(&args);
	kernel_launcher.setGroupSize(128, 1, 1);
	kernel_launcher.setGlobalSize(len, 1, 1);
}


void FirFilter::Run()
{
	kernel_launcher.LaunchKernel();
}


void FirFilter::Verify()
{
	for (int i = 0; i < len; i++)
	{
		printf("Output[%d]: %f\n", i, out[i]);
	}
}



