#ifndef HMM_H
#define HMM_H

#include "clUtil.h"

using namespace clHelper;

class HMM
{
	// Helper objects
	clRuntime *runtime;
	clFile *file;

	bool svmCoarseGrainAvail;
	bool svmFineGrainAvail;

	// OpenCL resources, auto release 
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_command_queue cmdQueue_0;
	//cl_command_queue cmdQueue_1;

	// User managed kernels, no auto release
	// Forward
	cl_kernel kernel_FWD_init_alpha;
	cl_kernel kernel_FWD_norm_alpha;
	cl_kernel kernel_TransposeSym;
	cl_kernel kernel_FWD_update_alpha;
	// Backward
	cl_kernel kernel_BK_BetaB;
	cl_kernel kernel_BK_update_beta;
	cl_kernel kernel_BK_norm_beta;
	// EM
	cl_kernel kernel_EM_betaB_alphabeta;
	cl_kernel kernel_EM_update_gamma;
	cl_kernel kernel_EM_alpha_betaB;
	cl_kernel kernel_EM_pre_xisum;
	cl_kernel kernel_EM_update_xisum;
	cl_kernel kernel_EM_gamma;
/*
	cl_kernel kernel_EM_sum_alphabeta;
	cl_kernel kernel_EM_alphabeta_update_gamma;
	cl_kernel kernel_EM_A_mul_alphabetaB;
	cl_kernel kernel_EM_norm_alphabeta;
	cl_kernel kernel_EM_alphabeta;
	cl_kernel kernel_EM_expt_A;
	cl_kernel kernel_EM_transpose;
	cl_kernel kernel_EM_gammastatesum;
	cl_kernel kernel_EM_gammaobs;
	cl_kernel kernel_EM_exptmu;
	cl_kernel kernel_EM_exptsigma_dev;
	cl_kernel kernel_EM_update_exptsigma;	
*/
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
	// Prepare
	float *a;          // state transition probability matrix
	float *b;          // emission probability matrix
	float *alpha;      // forward probability matrix
	float *prior;      // prior probability
	float *observations;

	// Forward
	float *lll;        // log likelihood
	float *aT;         // transpose of a

	// Backward 
	float *beta;
	float *betaB;

	// EM
	float *xi_sum;        // N x N
	float *alpha_beta;    // N
	float *gamma;         // T x N
	float *alpha_betaB;   // N x N
	float *xi_sum_tmp;    // N x N
	float *blk_result;    // intermediate blk results

	// Constant
	float *constMem;
/*

	// em 
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

*/
	//-------------------------------------------------------------------------------------------//
	// Initialize functions
	//-------------------------------------------------------------------------------------------//
	void Init();
	void InitParam();
	void InitCL();
	void InitKernels();
	void InitBuffers();

	//-------------------------------------------------------------------------------------------//
	// Clean functions
	//-------------------------------------------------------------------------------------------//
	void CleanUp();
	void CleanUpKernels();
	void CleanUpBuffers();

	//-------------------------------------------------------------------------------------------//
	// Forward functions
	//-------------------------------------------------------------------------------------------//
	void Forward();
	void ForwardInitAlpha();
	void ForwardNormAlpha(int startpos);
	void TransposeSym(float *a, float *aT, int size);
	void ForwardUpdateAlpha(int pos);

	//-------------------------------------------------------------------------------------------//
	// Backward functions
	//-------------------------------------------------------------------------------------------//
	void Backward();
	void BackwardBetaB(int pos);
	void BackwardUpdateBeta(int pos);
	void BackwardNormBeta(int pos);

	//-------------------------------------------------------------------------------------------//
	// EM functions
	//-------------------------------------------------------------------------------------------//
	void EM();
	void EM_betaB_alphabeta(int curpos, int prepos);
	void EM_update_gamma(int pos);
	void EM_alpha_betaB(int pos);
	void EM_pre_xisum();
	void EM_update_xisum(float sumvalue);
	void EM_gamma(int pos);




public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
