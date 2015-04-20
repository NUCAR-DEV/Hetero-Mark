#include <cstring>

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "IirFilter.h"

IirFilter::IirFilter() :
	kernel_launcher()
{
}


void IirFilter::Init() 
{
	// Init helper
	helper->Init();

	// Load kernels
	helper->LoadProgram("kernel.brig");

	// Setup kernel launcher
	kernel_launcher.setHelper(helper);
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
		helper->RegisterMemory(in, sizeof(float) * len);
	out = (float *)malloc(sizeof(float) * len * channels);
	helper->RegisterMemory(out, sizeof(float) * len *channels);
	for (int i = 0; i < len; i++)
	{
		in[i] = (float)i;
	}

	// Filter parameters
	nsec = (float *)malloc(sizeof(float) * rows * 2);
	helper->RegisterMemory(nsec, sizeof(float) * rows * 2);
	dsec = (float *)malloc(sizeof(float) * rows * 2);
	helper->RegisterMemory(dsec, sizeof(float) * rows * 2);
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
	args.sm = (void *)(4 * sizeof(float));

	// Set to kernel launcher
	kernel_launcher.setArguments(&args);
	kernel_launcher.setGroupSize(rows, 1, 1);
	kernel_launcher.setGlobalSize(channels * rows, 1, 1);
	kernel_launcher.setGroupSegmentSize(1024 * sizeof(float));
}


void IirFilter::Run()
{
	kernel_launcher.LaunchKernel();
}


void IirFilter::Verify()
{
	float *ds = (float*) malloc(sizeof(float) * rows * 2);	
	float *ns = (float*) malloc(sizeof(float) * rows * 2);	
	float *cpu_y = (float *) malloc(sizeof(float) * len);

	// internal state
	float *u = (float*) malloc(sizeof(float) * rows * 2);
	memset(u, 0 , sizeof(float) * rows * 2);

	float res, unew;
	for(int i = 0; i < rows; i++)
	{
		ds[i*2] = ds[i*2 + 1] = 0.00005f;
		ns[i*2] = ns[i*2 + 1] = 0.00002f;
	}

	for(int i=0; i<len; i++)
	{
		res = c * in[i];
		for(int j=0; j<rows; j++)
		{
			unew = in[i] - (ds[j*2] * u[j*2] + ds[j*2+1] * u[j*2+1]);
			u[j*2+1] = u[j * 2];
			u[j*2] = unew;
			res = res + (u[j*2] * ns[j*2] + u[j*2 + 1] * ns[j*2 + 1]);
		}

		cpu_y[i] = res;
	}

	// Compare result
	int success = 1;
	for(int chn=0; chn<channels; chn++)
	{
		size_t start = chn * len;

		for(int i = 0; i < len; i++)
		{
			if(abs(cpu_y[i] - out[i + start]) > 0.001)	
			{
				printf("Failed! Expect %f but was %f\n", 
						cpu_y[i], out[i + start]);
				success = 0;
			}
			else 
			{
				printf("Succeed! Expect %f and was %f\n",
						cpu_y[i], out[i+start]);
			}
		}
	}

	if(success)
		printf("Passed!\n");
}



