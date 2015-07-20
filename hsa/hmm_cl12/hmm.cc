#include <stdint.h>/* for uint64 definition */
#include <time.h>/* for clock_gettime */
#include <iostream>
#include <string.h>
#include <math.h>
#include <memory>
#include "hmm.h"
#include "kernels.h"

using namespace std;

#define BILLION 1000000000L

HMM::HMM(int N)
{
        if (N >= TILE)
                this->N = N;
        else
        {
                std::cout << "N < " << TILE << std::endl;
                exit(-1);
        }
}

HMM::~HMM()
{
        // Some cleanup
        CleanUp();
}

void HMM::Init()
{
        InitParam();
        InitBuffers();
}

void HMM::InitParam()
{
	T = 64;
	D = 64;
        if (N)
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
                bytes_const    = sizeof(float) * 4096; // 16 KB
                dd             = D * D;

                tileblks       = (N/TILE) * (N/TILE);// [N/16][N/16]
                bytes_tileblks = sizeof(float) * tileblks;

                blk_rows       = D/16;
                blknum         = blk_rows * (blk_rows + 1) / 2; 
        }
        else
        {
                std::cout << "Invalid N" << std::endl;
                exit(-1);
        }
}

void HMM::InitBuffers()
{

        // CPU buffers
        //-------------------------------------------------------------------------------------------//
        // SVM buffers 
        //        a,b,prior,lll, blk_result
        //-------------------------------------------------------------------------------------------//

        int i, j;

        //-------------------------------------------------------------------------------//
        // Prepare
        //-------------------------------------------------------------------------------//
        // state transition probability matrix
        a            = (float *)malloc( sizeof(float) * bytes_nn);

        // emission probability matrix 
        b            = (float *)malloc( sizeof(float) * bytes_nt);

        // forward probability matrix: TxN
        alpha       = (float *)malloc( sizeof(float) * bytes_nt);

        // prior probability
        prior       = (float *)malloc( sizeof(float) * bytes_n);

        // observed input 
        observations = (float *)malloc( sizeof(float) * bytes_dt);

        // Constant memory
        constMem     = (float *)malloc( sizeof(float) * bytes_const);

        //-------------------------------------------------------------------------------//
        // Forward 
        //-------------------------------------------------------------------------------//
        // log likelihood 
        lll = (float *) malloc(sizeof(float));

        aT  = (float *) malloc(bytes_nn * sizeof(float));

        //-------------------------------------------------------------------------------//
        // Backward 
        //-------------------------------------------------------------------------------//
        beta  = (float *) malloc(bytes_nt * sizeof(float));

        betaB = (float *) malloc(bytes_n * sizeof(float));


        //-------------------------------------------------------------------------------//
        // EM 
        //-------------------------------------------------------------------------------//
        xi_sum          = (float *) malloc(bytes_nn * sizeof(float));
        alpha_beta      = (float *) malloc(bytes_n * sizeof(float));
        gamma           = (float *) malloc(bytes_nt * sizeof(float));
        alpha_betaB     = (float *) malloc(bytes_nn * sizeof(float));
        xi_sum_tmp      = (float *) malloc(bytes_nn * sizeof(float));

        // intermediate blk results from the device
        blk_result      = (float *) malloc(bytes_tileblks * sizeof(float));

        // Expected values
        expect_prior    = (float *) malloc(bytes_n * sizeof(float));
        expect_A        = (float *) malloc(bytes_nn * sizeof(float));
        expect_mu       = (float *) malloc(bytes_dn * sizeof(float));
        expect_sigma    = (float *) malloc(bytes_ddn * sizeof(float));

        gamma_state_sum = (float *) malloc(bytes_n * sizeof(float));
        gamma_obs       = (float *) malloc(bytes_dt * sizeof(float));

   
	// Init value
        for (i = 0; i < (N * N); i++)
                a[i] = 1.0f/(float)N;
        for (i = 0; i < (N * T); i++)
                b[i] = 1.0f/(float)T;
        for (i = 0; i < N; i++)
                prior[i] = 1.0f/(float)N;
        for (i = 0 ; i < D; ++i)
                for (j = 0 ; j< T; ++j)
                        observations[i * T + j] = (float)j + 1.f; // D x T

}

void HMM::CleanUp()
{
        CleanUpBuffers();
}

void HMM::CleanUpBuffers()
{
        // CPU buffers
        free(a);
        free(b);
        free(alpha);
        free(prior);
        free(observations);

        // Forward 
        free(lll);
        free(aT);

        // Backward
        free(beta);
        free(betaB);

        // EM
        free(xi_sum);
        free(alpha_beta);
        free(gamma);
        free(alpha_betaB);
        free(xi_sum_tmp);
        free(blk_result);

        free(expect_prior);
        free(expect_A);
        free(expect_mu);
        free(expect_sigma);

        free(gamma_state_sum);
        free(gamma_obs);
}

        
//-----------------------------------------------------------------------------------------------//
//                                    Run Forward()
//-----------------------------------------------------------------------------------------------//
void HMM::Forward()
{
    // clear lll
    *lll = 0.f;

    ForwardInitAlpha();

    ForwardNormAlpha(0);

    TransposeSym(N);

    int frm;
    int current, previous;

    for (frm = 1; frm < T; ++frm) 
    {
        current  = frm * N; 
        previous = current - N;

	memcpy(constMem, alpha + previous, bytes_n);

        ForwardUpdateAlpha(current);

        // Normalize alpha at current frame
        // Update log likelihood
        ForwardNormAlpha(current);

    }
}

//-----------------------------------------------------------------------------------------------//
//                                    Run Backward()
//-----------------------------------------------------------------------------------------------//
void HMM::Backward()
{
    int j;
    int current, previous;    
    
    // TODO: xi_sum and gamma update could be run concurrently 
    for(j = T-2; j>=0; --j)    
    {
        current = j * N;        
        previous = current + N;

        // beta(t+1) .* b(t+1)
        BackwardBetaB(previous);

        // copy betaB to constant memory
	memcpy(constMem, betaB, bytes_n);

        // beta(t-1) = a * betaB
        BackwardUpdateBeta(current);    

        // normalize beta at current frame
        BackwardNormBeta(current);
    }
}


//-----------------------------------------------------------------------------------------------//
//                                    Run EM()
//-----------------------------------------------------------------------------------------------//

void HMM::EM()
{
    // clear data for xi_sum
    for (int i = 0; i < bytes_nn; i++)
    {
	    xi_sum[i] = 0.0f;
    }

    float sum;
    int i, current, previous;
    int window;

    for(window = 0; window < (T - 1); ++window)
    {
        current = window * N;   
        previous = current + N;

        // compute beta(t+1) * B(t+1) and alpha(t) * beta(t)
        EM_betaB_alphabeta(current, previous);

        // normalise alpha_beta, upate gamma for current frame
        EM_update_gamma(current);

        // alpha * betaB'
        EM_alpha_betaB(current);

        // normalise( A .*  (alpha * betaB') ) 
        // compute xi_sum_tmp and block results
        // FIXME: sometimes segmentation faults
        EM_pre_xisum();

        sum = 0.f;
        // compute on cpu
#pragma unroll
        for(i=0; i< tileblks; ++i)
            sum += blk_result[i];

        // Update the xi_sum
        EM_update_xisum(sum);
    }

    // update gamma at the last frame
    current = previous;

    // TODO: reuse template
    EM_gamma(current);

    //-------------------------------------------------------------------------------------------//
    // update expected values
    //-------------------------------------------------------------------------------------------//

    // TODO: Merge with previous gamma update
    // expected_prior = gamma(:, 1);
    
    // check from here !!!
    memcpy(expect_prior, gamma, bytes_n);

    // update expect_A
    EM_expectA();

    // compute gamma_state_sum
    // gamma is T x N, gamma_state_sum is the colum-wise summation
    EM_gamma_state_sum();

    // transpose observations


    // TODO: Concurrent Kernel Execution
    size_t start;
    int hs;
    for(hs = 0 ; hs < N; ++hs)
    {
        // copy gamma(hs,:) to constant memory
        memcpy(constMem, gamma + hs * T, bytes_t);

        // compute gammaobs
        EM_gamma_obs();

        current = hs * D;

        // compute expect_mu
        // TODO: map to host when there is not enough data?
        EM_expect_mu(current, hs);

        // copy expect_mu to constant mem
	memcpy(constMem, expect_mu + hs * D, bytes_d);

        // compute sigma_dev
        EM_sigma_dev(hs);


        start =  hs * dd;

        // update expect_sigma
        EM_expect_sigma(start);

    }


}










//-----------------------------------------------------------------------------------------------//
//                                        Forward Functions
//-----------------------------------------------------------------------------------------------//

void HMM::ForwardInitAlpha()
{
    size_t globalSize = (size_t)(ceil(N/256.f) * 256);
    size_t localSize = 256;

    SNK_INIT_LPARM(lparm, globalSize);
    lparm->ldims[0] = localSize;
    FWD_init_alphaKernel(N, b, prior, alpha, beta, lparm);
}

void HMM::ForwardNormAlpha(int startpos)
{
    size_t localSize = 256;
    size_t globalSize = (size_t)(ceil(N/256.f) * 256);

    SNK_INIT_LPARM(lparm, globalSize);
    lparm->ldims[0] = localSize;

    FWD_norm_alphaKernel(N, startpos, 256, alpha, lll, lparm);

}


void HMM::TransposeSym(int size)
{
	SNK_INIT_LPARM(lparm, 16);
	lparm->ndim = 2;
	lparm->gdims[0] = N;
	lparm->gdims[1] = N;
	lparm->ldims[0] = 16;
	lparm->ldims[1] = 16;

	TransposeSymKernel(N, 272, a, aT, lparm);
}

void HMM::ForwardUpdateAlpha(int pos)
{        
	int current = pos;

        SNK_INIT_LPARM(lparm,16);
	lparm->ndim=2;
	lparm->gdims[0]=16;
	lparm->gdims[1]=N;
	lparm->ldims[0]=16;
	lparm->ldims[1]=16;

	FWD_update_alphaKernel(N,current,272,constMem,aT,b,alpha,lparm);
}



//-----------------------------------------------------------------------------------------------//
//                                  Backward Functions
//-----------------------------------------------------------------------------------------------//

void HMM::BackwardBetaB(int pos)
{
    int previous = pos;    

    size_t globalSize = (size_t)(ceil(N/256.f) * 256);
    size_t localSize = 256;

    SNK_INIT_LPARM(lparm, globalSize);
    lparm->ldims[0] = localSize;
    BK_BetaBKernel(N, previous, beta, b, betaB, lparm);
}


void HMM::BackwardUpdateBeta(int pos)
{
    int current = pos;    

    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = 16;
    lparm->gdims[1] = N;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    BK_update_betaKernel(N, current, 272, constMem, a, beta, lparm);
}


void HMM::BackwardNormBeta(int pos)
{
	size_t localSize = 256;
	size_t globalSize = (size_t) (ceil(N/256.f) * 256);
	SNK_INIT_LPARM(lparm, globalSize);
	lparm->ldims[0] = localSize;
	int current = pos;
	BK_norm_betaKernel(N, current, 256, beta, lparm);

}


//-----------------------------------------------------------------------------------------------//
//                                        EM Functions
//-----------------------------------------------------------------------------------------------//

void HMM::EM_betaB_alphabeta(int curpos, int prepos)
{
    int current = curpos;    
    int previous = prepos;

    size_t localSize = 256;
    size_t globalSize = (size_t)(ceil(N/256.f) * 256);

	SNK_INIT_LPARM(lparm, globalSize);
	lparm->ldims[0] = localSize;
	EM_betaB_alphabetaKernel(N, current, previous, beta, b, alpha, betaB,
			alpha_beta, lparm);
}


void HMM::EM_update_gamma(int pos)
{

    int current = pos;    

    size_t localSize = 256;
    size_t globalSize = (size_t)(ceil(N/256.f) * 256);

	SNK_INIT_LPARM(lparm, globalSize);
	lparm->ldims[0] = localSize;
	EM_update_gammaKernel(N, current, 256, alpha_beta, gamma, lparm);
}


void HMM::EM_alpha_betaB(int pos)
{

    int current = pos;
    
    SNK_INIT_LPARM(lparm, 0);
    lparm->ndim = 2;
    lparm->gdims[0] = N;
    lparm->gdims[1] = N;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    EM_alpha_betaBKernel(N, current, betaB, alpha, alpha_betaB, lparm);
    
}

void HMM::EM_pre_xisum()
{
    SNK_INIT_LPARM(lparm, 0);
    lparm->ndim = 2;
    lparm->gdims[0] = N;
    lparm->gdims[1] = N;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    EM_pre_xisumKernel(N, 272, a, alpha_betaB, xi_sum_tmp, blk_result, lparm);
}

void HMM::EM_update_xisum(float sumvalue)
{    
    float sum = sumvalue;    

    SNK_INIT_LPARM(lparm, 0);
    lparm->ndim = 2;
    lparm->gdims[0] = N;
    lparm->gdims[1] = N;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;
    EM_update_xisumKernel(N, sum, xi_sum_tmp, xi_sum, lparm);
}

void HMM::EM_gamma(int pos)
{    
    int current = pos;    

    size_t localSize = 256;
    size_t globalSize = (size_t)(ceil(N/256.f) * 256);
	
    SNK_INIT_LPARM(lparm, globalSize);
    lparm->ldims[0] = localSize;

    EM_gammaKernel(N, current, 256, alpha, beta, gamma, lparm);
}



void HMM::EM_expectA()
{    
    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = 16;
    lparm->gdims[1] = N;
    lparm->ldims[0] = 16;
    lparm->ldims[0] = 16;

    EM_expectAKernel(N, 272, xi_sum, expect_A, lparm);
    
}

void HMM::EM_gamma_state_sum()
{
    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = N;
    lparm->gdims[1] = 16;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    EM_gamma_state_sumKernel(N, T, 272, gamma, gamma_state_sum, lparm);
}


void HMM::EM_gamma_obs()
{
	SNK_INIT_LPARM(lparm, 0);
	lparm->ndim = 2;
	lparm->gdims[0] = T;
	lparm->gdims[1] = D;
	lparm->ldims[0] = 16;
	lparm->ldims[1] = 16;

	EM_gamma_obsKernel(D, T, constMem, observations, gamma_obs, lparm);
}


void HMM::EM_expect_mu(int pos, int currentstate)
{
    int offset = pos;
    int hs = currentstate;

    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = 16;
    lparm->gdims[1] = D;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    EM_expect_muKernel(D, T, offset, hs, 272, gamma_obs, gamma_state_sum, 
		    expect_mu, lparm);
}


void HMM::EM_sigma_dev(int currentstate)
{
    int hs = currentstate;

    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = 16;
    lparm->gdims[1] = blknum;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    EM_sigma_devKernel(D, T, hs, constMem, gamma_obs, observations, 
		    gamma_state_sum, sigma_dev, lparm);
}


void HMM::EM_expect_sigma(size_t pos)
{
    // FIXME: no cast?
    int start = (int)pos;

    // FIXME: OCL 1.2 doesn't support non-uniform workgroup size, 
    // i.e global size must be multiples of local size
    // size_t localSize[2]  = {16, 16};
    size_t localSize[2]  = {16, (size_t)blknum};
    size_t globalSize[2] = {16, (size_t)blknum};

    SNK_INIT_LPARM(lparm, 16);
    lparm->ndim = 2;
    lparm->gdims[0] = globalSize[0];
    lparm->gdims[1] = globalSize[1];
    lparm->ldims[0] = localSize[0];
    lparm->ldims[1] = localSize[1];

    EM_expect_sigmaKernel(blk_rows, D, start, sigma_dev, expect_sigma, lparm); 
}


//-----------------------------------------------------------------------------------------------//
//                                      Run HMM 
//-----------------------------------------------------------------------------------------------//
void HMM::Run()
{

    //-------------------------------------------------------------------------------------------//
    // HMM Parameters
    //        a,b,prior,alpha
    //-------------------------------------------------------------------------------------------//
    printf("=>Initialize parameters.\n");
    Init();

    //-------------------------------------------------------------------------------------------//
    // Forward Algorithm on GPU 
    //-------------------------------------------------------------------------------------------//
    printf("\n");
    printf("      >> Start  Forward Algorithm on GPU.\n");
    Forward();
    printf("      >> Finish Forward Algorithm on GPU.\n");

    //-------------------------------------------------------------------------------------------//
    // Backward Algorithm on GPU 
    //-------------------------------------------------------------------------------------------//
    printf("\n");
    printf("      >> Start  Backward Algorithm on GPU.\n");
    Backward();
    printf("      >> Finish Backward Algorithm on GPU.\n");

    //-------------------------------------------------------------------------------------------//
    // EM Algorithm on GPU 
    //-------------------------------------------------------------------------------------------//
    printf("\n");
    printf("      >> Start  EM Algorithm on GPU.\n");
    EM();
    printf("      >> Finish EM Algorithm on GPU.\n");

    printf("<=End program.\n");

}

int main(int argc, char const *argv[])
{
  uint64_t diff;
  struct timespec start, end;
  if(argc != 2){
    puts("Please specify the number of hidden states N. (e.g., $./gpuhmmsr N)\nExit Program!");
    exit(1);
  }
  
  printf("=>Start program.\n");
  
  int N = atoi(argv[1]);
  
  // Smart pointer, auto cleanup in destructor
  std::unique_ptr<HMM> hmm(new HMM(N));

  
  //    double start = time_stamp();
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */
  hmm->Run();
  //double end = time_stamp();
  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("Total elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
  
  return 0;
}


