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
	cl_command_queue cmdQueue_0;
	cl_command_queue cmdQueue_1;

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

	static const int TILE = 16;
	static const int SIZE = 4096;
	int N;
	int T = 64;	// number of (overlapping) windows 
	int D = 64;	// number of features

	int bytes_nn; 
	int bytes_nt; 
	int bytes_dt;  
	int bytes_dd; 
	int bytes_dn; 
	int bytes_ddn; 
	int bytes_t;
	int bytes_d;  
	int bytes_n;  
	int dd;

	int tileblks;
	size_t bytes_tileblks;

	float *a;          // state transition probability matrix
	float *b;          // emission probability matrix
	float *pi;         // prior probability
	float *alpha;      // forward probability matrix
	float *lll;        // log likelihood
	float *blk_result; // intermediate blk results
	float *observations;

	void Init();
	void InitParam();
	void InitCL();

	void InitKernels();
	void InitBuffers();

	void CleanUp();
	void CleanUpKernels();
	void CleanUpBuffers();

	void Forward();
	void Backward();
	void BaumWelch();

public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
