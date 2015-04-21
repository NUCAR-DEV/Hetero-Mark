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

	// Init arguments
	InitParam();

	// Init Buffers;
	InitBuffers();
}


void HMM::InitKernels()
{
	// Load program
	helper->LoadProgram("kernels.brig");

	// Setup kernel launchers
	fwd_init_alpha.reset(new FwdInitAlphaHsaLauncher(helper));
	fwd_init_alpha->Init();
	fwd_norm_alpha.reset(new FwdNormAlphaHsaLauncher(helper));
	fwd_norm_alpha->Init();
	transpose_sym.reset(new TransposeSymHsaLauncher(helper));
	transpose_sym->Init();
	fwd_update_alpha.reset(new FwdUpdateAlphaHsaLauncher(helper));
	fwd_update_alpha->Init();

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

	// Backward
	beta = helper->CreateBuffer(bytes_nt);
	betaB = helper->CreateBuffer(bytes_n);

	// EM
	xi_sum = helper->CreateBuffer(bytes_nn);
	alpha_beta = helper->CreateBuffer(bytes_n);
	gamma = helper->CreateBuffer(bytes_nt);
	alpha_betaB = helper->CreateBuffer(bytes_nn);
	xi_sum_tmp = helper->CreateBuffer(bytes_nn);

	// intermediate blk results from the device
	blk_result      = helper->CreateBuffer(bytes_tileblks);

	// Expected values
	expect_prior    = helper->CreateBuffer(bytes_n);
	expect_A        = helper->CreateBuffer(bytes_nn);
	expect_mu       = helper->CreateBuffer(bytes_dn);
	expect_sigma    = helper->CreateBuffer(bytes_ddn);

	gamma_state_sum = helper->CreateBuffer(bytes_n);
	gamma_obs       = helper->CreateBuffer(bytes_dt);

	// Setup init value
	timer->BeginTimer();
	a_host = (float *)malloc(sizeof(float) * N * N);
	b_host = (float *)malloc(sizeof(float) * N * T);
	prior_host = (float *)malloc(sizeof(float) * N);
	observation_host = (float *)malloc(sizeof(float) * D * T);
	for (int i = 0; i < (N * N); i++)
	{
		a_host[i] = 1.0f/(float)N;
	}
	for (int i = 0; i < (N * T); i++)
	{
		b_host[i] = 1.0f/(float)T;
	}
	for (int i = 0; i < (N); i++)
	{
		prior_host[i] = 1.0f/(float)N;
	}
	for (int i = 0; i < D; i++)
	{
		for(int j = 0; j < T; j++)
		{
			observation_host[i * T + j] = (float)j + 1.f;
		}
	}
	timer->EndTimer({"CPU", "memory"});
	memcpy(a, a_host, sizeof(float) * N * N);
	memcpy(b, b_host, sizeof(float) * N * T);
	memcpy(prior, prior_host, sizeof(float) * N);
	memcpy(observations, observation_host, sizeof(float) * D * T);
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
	*(float *)lll = 0.f;
	ForwardInitAlpha();
	ForwardNormAlpha(0);
	TransposeSym((float *)a, (float *)aT, N);

	int frm;
	int current, previous;
	for (frm = 1; frm < T; frm++)
	{
		current = frm * N;
		previous = current - N;

		timer->BeginTimer();
		memcpy(constMem, &(((float *)alpha)[previous]), bytes_n);
		timer->EndTimer({"CPU", "memory"});

		ForwardUpdateAlpha(current);
		ForwardNormAlpha(current);
	}
}


void HMM::ForwardInitAlpha()
{
#if 0
	for (int i = 0; i < N; i++)
	{
		((float *)beta)[i] = 1.0;
		((float *)alpha)[i] = ((float *)prior)[i] * ((float *)b)[i];
	}
#else
	// Set global and local size
	fwd_init_alpha->setGlobalSize(ceil(N/256.f)*256, 1, 1);
	fwd_init_alpha->setGroupSize(256, 1, 1);

	// Set Arguments
	fwd_init_alpha->setArgument(0, sizeof(int), &N);
	fwd_init_alpha->setArgument(1, sizeof(uint64_t), &b);
	fwd_init_alpha->setArgument(2, sizeof(uint64_t), &prior);
	fwd_init_alpha->setArgument(3, sizeof(uint64_t), &alpha);
	fwd_init_alpha->setArgument(4, sizeof(uint64_t), &beta);

	// Launch Kernel
	fwd_init_alpha->LaunchKernel();
#endif
}


void HMM::ForwardNormAlpha(int start_pos)
{
	// Set size
	fwd_norm_alpha->setGlobalSize(ceil(N/256.f)*256, 1, 1);
	fwd_norm_alpha->setGroupSize(256, 1, 1);

	// Set arguments
	int pos = start_pos;
	fwd_norm_alpha->setArgument(0, sizeof(int), (void *)&N);
	fwd_norm_alpha->setArgument(1, sizeof(int), (void *)&pos);
	fwd_norm_alpha->setArgument(2, sizeof(float)*256, NULL);
	fwd_norm_alpha->setArgument(3, sizeof(void *), (void *)&alpha);
	fwd_norm_alpha->setArgument(4, sizeof(void *), (void *)&lll);

	// Launch Kernel
	fwd_norm_alpha->LaunchKernel();
}


void HMM::TransposeSym(float *a, float *aT, int size)
{
	// Set dim and size
	transpose_sym->setDimension(2);
	transpose_sym->setGroupSize(16, 16, 1);
	transpose_sym->setGlobalSize(N, N, 1);

	// Set arguments
	transpose_sym->setArgument(0, sizeof(int), (void *)&N);
	transpose_sym->setArgument(1, sizeof(float) * 272, NULL);
	transpose_sym->setArgument(2, sizeof(void *), (void *)&a);
	transpose_sym->setArgument(3, sizeof(void *), (void *)&aT);

	// Launch kernel
	transpose_sym->LaunchKernel();
}


void HMM::ForwardUpdateAlpha(int pos)
{
	int current = pos;

	// Set dim and size
	fwd_update_alpha->setDimension(2);
	fwd_update_alpha->setGroupSize(16, 16, 1);
	fwd_update_alpha->setGlobalSize(16, N, 1);

	// Set argument
	fwd_update_alpha->setArgument(0, sizeof(int), &N);
	fwd_update_alpha->setArgument(1, sizeof(int), &current);
	fwd_update_alpha->setArgument(2, sizeof(float) * 272, NULL);
	fwd_update_alpha->setArgument(3, sizeof(void *), (void *)&constMem);
	fwd_update_alpha->setArgument(4, sizeof(void *), (void *)&aT);
	fwd_update_alpha->setArgument(5, sizeof(void *), (void *)&b);
	fwd_update_alpha->setArgument(6, sizeof(void *), (void *)alpha);

	// LaunchKernel
	fwd_update_alpha->LaunchKernel();

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



