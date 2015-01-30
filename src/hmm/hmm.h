#ifndef HMM_H
#define HMM_H

#include "clUtil.h"

using namespace clHelper;

class HMM
{
	// Helper objects
	clRuntime *runtime;
	clFile *file;

	// OpenCL resources, auto release 
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_command_queue cmdQueue_0;
	cl_command_queue cmdQueue_1;

	// User managed kernels, no auto release
	cl_kernel kernel_FWD_init_alpha;
	cl_kernel kernel_FWD_scaling;
	cl_kernel kernel_FWD_calc_alpha;
	cl_kernel kernel_FWD_sum_ll;
	cl_kernel kernel_BK_update_beta;
	cl_kernel kernel_BK_scaling;
	cl_kernel kernel_EM_betaB_alphabeta;
	cl_kernel kernel_EM_sum_alphabeta;
	cl_kernel kernel_EM_alphabeta_update_gamma;
	cl_kernel kernel_EM_A_mul_alphabetaB;
	cl_kernel kernel_EM_update_xisum;
	cl_kernel kernel_EM_norm_alphabeta;
	cl_kernel kernel_EM_alphabeta;
	cl_kernel kernel_EM_expt_A;
	cl_kernel kernel_EM_transpose;
	cl_kernel kernel_EM_gammastatesum;
	cl_kernel kernel_EM_gammaobs;
	cl_kernel kernel_EM_exptmu;
	cl_kernel kernel_EM_exptsigma_dev;
	cl_kernel kernel_EM_update_exptsigma;	

	// Parameters
	static const int TILE = 16;
	static const int SIZE = 4096;
	static const int BLOCKSIZE = 256;

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
	int bytes_const;
	int dd;

	int tileblks;
	size_t bytes_tileblks;

	int blk_rows;
	int blknum;

	// SVM buffers, no auto release
	float *a;          // state transition probability matrix
	float *b;          // emission probability matrix
	float *prior;      // prior probability
	float *alpha;      // forward probability matrix
	float *lll;        // log likelihood
	float *blk_result; // intermediate blk results
	float *observations;

	// forward
	float *ones;
	float *ll;

	// bk
	float *beta;
	float *betaB;

	// em 
	float *xi_sum;
	float *alpha_beta;
	float *gamma;
	float *A_alphabetaB;
	float *gammaT;
	float *gamma_state_sum;
	float *gamma_obs; // D x T

	// expected values
	float *expt_prior;
	float *expt_A;
	float *observationsT;
	float *expt_mu; // N x D
	float *expt_sigma_sym;
	float *expt_sigma;

	// constant memory
	// hint: reuse constant buffer if possible
	float *constA; 
	float *constB;
	float *gamma_state_sumC;
	float *constT;
	float *expt_mu_state;

	void Init();
	void InitParam();
	void InitCL();

	void InitKernels();
	void InitBuffers();

	void CleanUp();
	void CleanUpKernels();
	void CleanUpBuffers();

	void Forward();
	void ForwardInitAlpha(int numElements, float *bSrc, float *piSrc, 
		float *alphaDst, float *onesDst, float *betaDst);
	void ForwardSumAlpha();
	void ForwardScaling(int numElements, float *scaleArraySrc, int scaleArrayIndexSrc, float *dataDst);
	void ForwardCalcAlpha(int numElements, float *bSrc, float *alphaDst);
	void ForwardSumLL(int numElements, float *llDst);

	void Backward();
	void BackwardUpdateBeta(int numElements, float *betaSrc, float *bSrc, float *betaBDst);
	void BackwardScaling(int numElements, float *llSrc, float *betaDst);

	void BaumWelch();

public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
