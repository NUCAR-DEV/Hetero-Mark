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
	int dd;

	int tileblks;
	size_t bytes_tileblks;

	// SVM buffers, no auto release
	float *a;          // state transition probability matrix
	float *b;          // emission probability matrix
	float *pi;         // prior probability
	float *alpha;      // forward probability matrix
	float *lll;        // log likelihood
	float *blk_result; // intermediate blk results
	float *observations;

	// forward
	float *ones_d;
	float *ll_d;

	// bk
	float *beta_d;
	float *betaB_d;

	// em 
	float *xi_sum_d;
	float *alpha_beta_d;
	float *gamma_d;
	float *A_alphabetaB_d;
	float *gammaT_d;
	float *gamma_state_sum_d;
	float *gamma_obs_d;
	float *expect_mu_d;
	float *expect_sigma_sym_d;
	float *expect_sigma_d;

	float *expect_prior_d;
	float *expect_A_d;
	float *observationsT_d;


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
	void ForwardScaling(int numElements, float *scaleArraySrc, 
		int scaleArrayIndexSrc, float *dataDst);
	void ForwardCalcAlpha(int numElements, float *alpha, float *b);

	void Backward();
	void BackwardUpdateBeta(int numElements, float *betaSrc, float *bSrc, float *betaBDst);
	void BackwardScaling(int numElements, float *llSrc, float *betaDst);

	void BaumWelch();
	void EMBetaBAlphaBeta(int numElements, int curWindow, int preWindow, 
		float *betaSrc, float *BSrc, float *alphaSrc, float *betaBDst, float *alphaBetaDst);
	void EMAlphaBetaUpdateGamma(int numElements, int curWindow, float *alphaBetaSrc,
		float *llSrc, float *gammaDst);
	void EMAMulAlphaBetaB(int numElements, float *ASrc, float *AAlphaBetaBDst, 
		float *blkResultDst, float *constA, float *constB);
	void EMSumBlkresult(float *sum);
	void EMUpdateXisum(int numElements, float sum, float *AAlphaBetaBSrc, float *xiSumDst);
	void EMAlphaBeta(int numElements, float *alphaSrc, float *betaSrc, float *alphaBetaDst);
	void EMExpectA(int numElements, float *xiSumSrc, float *expectADst);

public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
