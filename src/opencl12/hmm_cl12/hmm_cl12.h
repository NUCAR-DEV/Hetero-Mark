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
	cl_kernel kernel_EM_expectA;
	cl_kernel kernel_EM_gamma_state_sum;
	cl_kernel kernel_EM_gamma_obs;
	cl_kernel kernel_EM_expect_mu;
	cl_kernel kernel_EM_sigma_dev;
	cl_kernel kernel_EM_expect_sigma;	

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

	// OCL 1.2 buffers
	// Prepare
	cl_mem a;            // state transition probability matrix
	cl_mem b;            // emission probability matrix
	cl_mem alpha;        // forward probability matrix
	cl_mem prior;        // prior probability
	cl_mem observations; // D x T

	// Forward
	cl_mem lll;        // log likelihood
	cl_mem aT;         // transpose of a

	// Backward 
	cl_mem beta;
	cl_mem betaB;

	// EM
	cl_mem xi_sum;          // N x N
	cl_mem alpha_beta;      // N
	cl_mem gamma;           // T x N
	cl_mem alpha_betaB;     // N x N
	cl_mem xi_sum_tmp;      // N x N
	cl_mem blk_result;      // intermediate blk results

	cl_mem expect_prior;    // N
	cl_mem expect_A;        // N xN
	cl_mem expect_mu;       // N x D
	cl_mem expect_sigma;    // N x D x D

	cl_mem gamma_state_sum; // N
	cl_mem gamma_obs;       // D x T
	cl_mem sigma_dev;       // D x D

	// Constant
	cl_mem constMem;



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
	void TransposeSym(int size);
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
	void EM_expectA();
	void EM_gamma_state_sum();
	void EM_gamma_obs();
	void EM_expect_mu(int pos, int currentstate);
	void EM_sigma_dev(int currentstate);
	void EM_expect_sigma(size_t pos);

public:
	HMM(int N);
	~HMM();

	void Run();

};

#endif
