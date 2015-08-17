#ifndef HMM_H
#define HMM_H

#define __constant const

class HMM
{
	// Parameters
	static const int TILE = 16;
	static const int SIZE = 4096;
	static const int BLOCKSIZE = 256;

	int N;
	int T;	// number of (overlapping) windows 
	int D;	// number of features

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
	float *a;            // state transition probability matrix
	float *b;            // emission probability matrix
	float *alpha;        // forward probability matrix
	float *prior;        // prior probability
	float *observations; // D x T

	// Forward
	float *lll;        // log likelihood
	float *aT;         // transpose of a

	// Backward 
	float *beta;
	float *betaB;

	// EM
	float *xi_sum;          // N x N
	float *alpha_beta;      // N
	float *gamma;           // T x N
	float *alpha_betaB;     // N x N
	float *xi_sum_tmp;      // N x N
	float *blk_result;      // intermediate blk results

	float *expect_prior;    // N
	float *expect_A;        // N xN
	float *expect_mu;       // N x D
	float *expect_sigma;    // N x D x D

	float *gamma_state_sum; // N
	float *gamma_obs;       // D x T
	float *sigma_dev;       // D x D

	// Constant
	float *constMem;



	//-------------------------------------------------------------------------------------------//
	// Initialize functions
	//-------------------------------------------------------------------------------------------//
	void Init();
	void InitParam();
	void InitBuffers();

	//-------------------------------------------------------------------------------------------//
	// Clean functions
	//-------------------------------------------------------------------------------------------//
	void CleanUp();
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
