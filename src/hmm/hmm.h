#ifndef HMM_H
#define HMM_H

#include "clUtil.h"

using namespace clHelper;

class HMM
{
	clRuntime *runtime;
	clFile *file;

	cl_device_id device;
	cl_context context;
	cl_program program;

	cl_kernel kernel_FWD_init_alpha;
	cl_kernel kernel_FWD_scaling;
	cl_kernel kernel_FWD_calc_alpha;
	cl_kernel kernel_FWD_sum_ll;
	cl_kernel kernel_BK_update_beta;
	cl_kernel kernel_BK_scaling;
	cl_kernel kernel_EM_betaB_alphabeta;
	cl_kernel kernel_EM_alphabeta_update_gamma;
	cl_kernel kernel_EM_A_mul_alphabetaB;
	cl_kernel kernel_EM_update_xisum;
	cl_kernel kernel_EM_alphabeta;
	cl_kernel kernel_EM_expect_A;
	cl_kernel kernel_EM_transpose;
	cl_kernel kernel_EM_gammastatesum;
	cl_kernel kernel_EM_gammaobs;
	cl_kernel kernel_EM_expectmu;
	cl_kernel kernel_EM_expectsigma_dev;
	cl_kernel kernel_EM_update_expectsigma;	

	int N;

	void SetupCL();
	void Param();
	void Forward();
	void Backward();
	void BaumWelch();

	void Release();
public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
