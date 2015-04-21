#ifndef HMM_CL12_HMM_H
#define HMM_CL12_HMM_H

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"
#include "../common/Benchmark.h"

#include "FwdInitAlphaHsaLauncher.h"
#include "FwdNormAlphaHsaLauncher.h"
#include "TransposeSymHsaLauncher.h"
#include "FwdUpdateAlphaHsaLauncher.h"

class KernelLauncher;

/**
 * IirFilter benchmark
 */
class HMM : public Benchmark 
{
	// Dataset
	static const int TILE = 16;
	static const int SIZE = 4096;
	static const int BLOCKSIZE = 256;

	int N;
	int T = 64;
	int D = 64;

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

	// Prepare
	void *a;         
	void *b;          
	void *alpha;      
	void *prior;      
	void *observations;

	// Forward
	void *lll;      
	void *aT;        

	// Backward 
	void *beta;
	void *betaB;

	// EM
	void *xi_sum;      
	void *alpha_beta;   
	void *gamma;         
	void *alpha_betaB;    
	void *xi_sum_tmp;      
	void *blk_result;      

	void *expect_prior;    
	void *expect_A;        
	void *expect_mu;       
	void *expect_sigma;    

	void *gamma_state_sum; 
	void *gamma_obs;       
	void *sigma_dev;       

	// Constant
	void *constMem;

	// Memory in the host
	float *a_host;
	float *b_host;
	float *prior_host;
	float *observation_host;

	// Kernel Launchers
	std::unique_ptr<KernelLauncher> fwd_init_alpha;
	std::unique_ptr<KernelLauncher> fwd_norm_alpha;
	std::unique_ptr<KernelLauncher> transpose_sym;
	std::unique_ptr<KernelLauncher> fwd_update_alpha;

	// Init params
	void InitParam();

	// Init buffers
	void InitBuffers();

	// Init kernels
	void InitKernels();




	//
	// Forward Algorithm
	//

	// Forward
	void Forward();

	// Forward Init Alpha
	void ForwardInitAlpha();

	// Forward Norm Alpha
	void ForwardNormAlpha(int start_pos);

	// Transpose
	void TransposeSym(float *a, float *aT, int size);

	// Forward update alpha
	void ForwardUpdateAlpha(int pos);

	public:

	/**
	 * Constructor
	 */
	HMM();

	/**
	 * Init
	 */
	void Init() override;

	/**
	 * Run
	 */
	void Run() override;

	/**
	 * Verify
	 */
	void Verify() override;

	/**
	 * Summarize
	 */
	void Summarize() override;

	/**
	 * Set data length
	 */
	void setNumHiddenState(uint32_t N) 
	{ 
		if (N < TILE)
		{
			printf("N is smaller than %d.\n", TILE);
			exit(1);
		}
		this->N = N; 
	}
};

#endif
