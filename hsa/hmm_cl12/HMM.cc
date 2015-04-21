#include <cstring>
#include <math.h>

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "HMM.h"

HMM::HMM() :
	Benchmark()
{
}


void HMM::Init() 
{
	// Init helper
	helper->Init();

	// Init kernels
	InitKernels();

	// Setup kernel launcher
	/*
	kernel_launcher.setHelper(helper);
	kernel_launcher.setName("&__OpenCL_ParIIR_kernel");
	kernel_launcher.Init();
	*/

	// Init arguments
	InitParam();

	// Init Buffers;
	InitBuffers();
}


void HMM::InitKernels()
{
	// Load program
	helper->LoadProgram("kernels.brig");
}


void HMM::InitBuffers()
{
	// Prepare
	a = helper->CreateBuffer(bytes_nn);
	b = helper->CreateBuffer(bytes_nt);
	alpha = helper->CreateBuffer(bytes_nt);
	prior = helper->CreateBuffer(bytes_n);
	observations = helper->CreateBuffer(bytes_dt);
	constMem = helper->CreateBuffer(bytes_const);

	// Forward
	lll = helper->CreateBuffer(sizeof(float));
	aT = helper->CreateBuffer(bytes_nn);
}


void HMM::InitParam()
{
	bytes_nn       = sizeof(float) * N * N;
	bytes_nt       = sizeof(float) * N * T;
	bytes_n        = sizeof(float) * N;
	bytes_dt       = sizeof(float) * D * T;
	bytes_dd       = sizeof(float) * D * D;
	bytes_dn       = sizeof(float) * D * N ;
	bytes_ddn      = sizeof(float) * D * D * N ;
	bytes_t        = sizeof(float) * T;
	bytes_d        = sizeof(float) * D;
	bytes_n        = sizeof(float) * N;
	bytes_const    = sizeof(float) * 4096;
	dd             = D * D;
	tileblks       = (N/TILE) * (N/TILE);
	bytes_tileblks = sizeof(float) * tileblks;
	blk_rows       = D/16;
	blknum         = blk_rows * (blk_rows + 1) / 2; 
}




//
// Forward Algorithm
//

void HMM::Forward()
{
	float zero = 0.f;
	ForwardInitAlpha();
}


void HMM::ForwardInitAlpha()
{
	// Set global and local size
	fwd_init_alpha->setGlobalSize(ceil(N/256.f)*256, 1, 1);
	fwd_init_alpha->setGroupSize(256, 1, 1);

	// Set Arguments
	fwd_init_alpha->setArgument(0, sizeof(int), &N);
	fwd_init_alpha->setArgument(1, sizeof(void *), &b);
	fwd_init_alpha->setArgument(2, sizeof(void *), &prior);
	fwd_init_alpha->setArgument(3, sizeof(void *), &alpha);
	fwd_init_alpha->setArgument(4, sizeof(void *), &beta);

	// Launch Kernel
	fwd_init_alpha->LaunchKernel();
	
}


void HMM::Run()
{
	Forward();
}


void HMM::Verify()
{
}


void HMM::Summarize()
{
	printf("HMM benchmark: \n\n");
	timer->Summarize();
}



