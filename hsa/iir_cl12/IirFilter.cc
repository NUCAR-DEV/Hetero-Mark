#include <cstring>

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "IirFilter.h"

IirFilter::IirFilter() :
	helper(),
	kernel_launcher()
{
}


void IirFilter::Init() 
{
	// Init helper
	helper.Init();

	// Load kernels
	helper.LoadProgram("kernel.brig");

	// Setup kernel launcher
	kernel_launcher.setHelper(&helper);
	kernel_launcher.setName("&__OpenCL_ParIIR_kernel");
	kernel_launcher.Init();

	// Init arguments
	InitParam();
}


void IirFilter::InitParam()
{
	// Reset
	memset(&args, 0, sizeof(args));

	// Set argument
	channels = 64;
	c = 3.0f;
	rows = 256;

	// Input and output argument
	in = (float *)malloc(sizeof(float) * len);
	out = (float *)malloc(sizeof(float) * len * channels);
	for (int i = 0; i < len; i++)
	{
		in[i] = i;
	}

	// Filter parameters
	nsec = (float *)malloc(sizeof(float) * rows * 2);
	dsec = (float *)malloc(sizeof(float) * rows * 2);
	for (int i = 0; i < rows; i++)
	{
		nsec[2 * i] = 0.00002f;
		nsec[2 * i + 1] = 0.00002f;
		dsec[2 * i] = 0.00005f;
		dsec[2 * i + 1] = 0.00005f;
	}
	
	// Write args	
	args.x = in;
	args.y = out;
	args.nsec = nsec;
	args.dsec = dsec;
	args.len = len;
	args.c = c;
	args.sm = 0;

	// Set to kernel launcher
	kernel_launcher.setArguments(&args);
	kernel_launcher.setGroupSize(rows, 1, 1);
	kernel_launcher.setGlobalSize(channels * rows, 1, 1);
	kernel_launcher.setGroupSegmentSize(512 * sizeof(float));

}


void IirFilter::Run()
{
	kernel_launcher.LaunchKernel();
}


void IirFilter::Verify()
{
	for (int i = 0; i < len; i++)
	{
		printf("in[%d] = %f\n", i, out[i]);
	}
}


