#ifndef HMM_CL12_HMM_H
#define HMM_CL12_HMM_H

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"
#include "../common/Benchmark.h"

#include "FwdInitAlphaHsaLauncher.h"

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

	// Kernel Launchers
	std::unique_ptr<KernelLauncher> fwd_init_alpha;

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
	void setNumHiddenState(uint32_t N) { this->N = N; }
};

#endif
