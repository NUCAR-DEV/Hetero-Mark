#include <cstring>

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "IirKernelHsaLauncher.h"
#include "IirFilter.h"

IirFilter::IirFilter() :
	Benchmark()
{
}


void IirFilter::Init() 
{
	// Init helper
	helper->Init();

	// Init arguments
	InitParam();

	// Init Kernels
	InitKernels();
}


void IirFilter::InitKernels() 
{	
	// Load kernels
	helper->LoadProgram("kernel.brig");

	// Kernel
	iir_kernel.reset(new IirKernelHsaLauncher(helper));

	// Setup kernel launcher
	iir_kernel->Init();
}


void IirFilter::InitParam()
{
	// Set argument
	channels = 64;
	c = 3.0f;
	rows = 256;

	// Input and output argument
	in = (float *)helper->CreateBuffer(sizeof(float) * len);
	out = (float *)helper->CreateBuffer(sizeof(float) * len * channels);
	
	// Init Input
	timer->BeginTimer();
	for (int i = 0; i < len; i++)
	{
		in[i] = (float)i;
	}
	timer->EndTimer({"CPU"});

	// Filter parameters
	nsec = (float *)helper->CreateBuffer(sizeof(float) * rows * 2);
	dsec = (float *)helper->CreateBuffer(sizeof(float) * rows * 2);
	timer->BeginTimer();
	for (int i = 0; i < rows; i++)
	{
		nsec[2 * i] = 0.00002f;
		nsec[2 * i + 1] = 0.00002f;
		dsec[2 * i] = 0.00005f;
		dsec[2 * i + 1] = 0.00005f;
	}
	timer->EndTimer({"CPU"});

	// Write args	

}


void IirFilter::Run()
{
	// Set kernel size
	iir_kernel->setGroupSize(rows, 1, 1);
	iir_kernel->setGlobalSize(channels * rows, 1, 1);

	// Set argument
	iir_kernel->setArgument(0, sizeof(uint32_t), &len);
	iir_kernel->setArgument(1, sizeof(float), &c);
	iir_kernel->setArgument(2, sizeof(void *), &nsec);
	iir_kernel->setArgument(3, sizeof(void *), &dsec);
	iir_kernel->setArgument(4, sizeof(void *), NULL);
	iir_kernel->setArgument(5, sizeof(void *), &in);
	iir_kernel->setArgument(6, sizeof(void *), &out);

	// Launch kernel
	iir_kernel->LaunchKernel();
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


void IirFilter::Summarize()
{
	printf("IIR Filter benchmark: \n\n");
	timer->Summarize();
}



