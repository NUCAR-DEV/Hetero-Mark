/* HEADER FILE GENERATED BY snack VERSION 0.9.2 */
/* THIS FILE:  /home/yifan/hsa/hsaBench/hsa/hmm_cl12/kernels.h  */
/* INPUT FILE: /home/yifan/hsa/hsaBench/hsa/hmm_cl12/kernels.cl  */
#ifdef __cplusplus
#define _CPPSTRING_ "C" 
#endif
#ifndef __cplusplus
#define _CPPSTRING_ 
#endif
#ifndef __SNK_DEFS
#define SNK_MAX_STREAMS 8 
extern _CPPSTRING_ void stream_sync(const int stream_num);

#define SNK_ORDERED 1
#define SNK_UNORDERED 0

#include <stdint.h>
#ifndef HSA_RUNTIME_INC_HSA_H_
typedef struct hsa_signal_s { uint64_t handle; } hsa_signal_t;
#endif

typedef struct snk_task_s snk_task_t;
struct snk_task_s { 
   hsa_signal_t signal ; 
   snk_task_t* next;
};

typedef struct snk_lparm_s snk_lparm_t;
struct snk_lparm_s { 
   int ndim;                  /* default = 1 */
   size_t gdims[3];           /* NUMBER OF THREADS TO EXECUTE MUST BE SPECIFIED */ 
   size_t ldims[3];           /* Default = {64} , e.g. 1 of 8 CU on Kaveri */
   int stream;                /* default = -1 , synchrnous */
   int barrier;               /* default = SNK_UNORDERED */
   int acquire_fence_scope;   /* default = 2 */
   int release_fence_scope;   /* default = 2 */
} ;

/* This string macro is used to declare launch parameters set default values  */
#define SNK_INIT_LPARM(X,Y) snk_lparm_t * X ; snk_lparm_t  _ ## X ={.ndim=1,.gdims={Y},.ldims={64},.stream=-1,.barrier=SNK_UNORDERED,.acquire_fence_scope=2,.release_fence_scope=2} ; X = &_ ## X ;
 
/* Equivalent host data types for kernel data types */
typedef struct snk_image3d_s snk_image3d_t;
struct snk_image3d_s { 
   unsigned int channel_order; 
   unsigned int channel_data_type; 
   size_t width, height, depth;
   size_t row_pitch, slice_pitch;
   size_t element_size;
   void *data;
};

#define __SNK_DEFS
#endif
extern _CPPSTRING_ void FWD_init_alphaKernel(const int N,float* b,float* prior,float* alpha,float* beta, const snk_lparm_t * lparm);
extern _CPPSTRING_ void FWD_init_alphaKernel_init(const int printStats);
extern _CPPSTRING_ void FWD_norm_alphaKernel(const int N,const int startpos,size_t sm_size,float* alpha,float* lll, const snk_lparm_t * lparm);
extern _CPPSTRING_ void FWD_norm_alphaKernel_init(const int printStats);
extern _CPPSTRING_ void TransposeSymKernel(const int N,size_t sm_size,float* a,float* aT, const snk_lparm_t * lparm);
extern _CPPSTRING_ void TransposeSymKernel_init(const int printStats);
extern _CPPSTRING_ void FWD_update_alphaKernel(const int N,const int current,size_t sm_size,__constant float* constMem,float* aT,float* b,float* alpha, const snk_lparm_t * lparm);
extern _CPPSTRING_ void FWD_update_alphaKernel_init(const int printStats);
extern _CPPSTRING_ void BK_BetaBKernel(const int N,const int pos,const float* beta,const float* b,float* betaB, const snk_lparm_t * lparm);
extern _CPPSTRING_ void BK_BetaBKernel_init(const int printStats);
extern _CPPSTRING_ void BK_update_betaKernel(const int N,const int current,size_t sm_size,__constant float* constMem,float* a,float* beta, const snk_lparm_t * lparm);
extern _CPPSTRING_ void BK_update_betaKernel_init(const int printStats);
extern _CPPSTRING_ void BK_norm_betaKernel(const int N,const int current,size_t sm_size,float* beta, const snk_lparm_t * lparm);
extern _CPPSTRING_ void BK_norm_betaKernel_init(const int printStats);
extern _CPPSTRING_ void EM_betaB_alphabetaKernel(const int N,const int current,const int previous,const float* beta,const float* b,const float* alpha,float* betaB,float* alpha_beta, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_betaB_alphabetaKernel_init(const int printStats);
extern _CPPSTRING_ void EM_update_gammaKernel(const int N,const int current,size_t sm_size,const float* alpha_beta,float* gamma, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_update_gammaKernel_init(const int printStats);
extern _CPPSTRING_ void EM_alpha_betaBKernel(const int N,const int current,const float* betaB,const float* alpha,float* alpha_betaB, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_alpha_betaBKernel_init(const int printStats);
extern _CPPSTRING_ void EM_pre_xisumKernel(const int N,size_t sm_size,const float* a,const float* alpha_betaB,float* xi_sum_tmp,float* blk_result, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_pre_xisumKernel_init(const int printStats);
extern _CPPSTRING_ void EM_update_xisumKernel(const int N,const float sum,const float* xi_sum_tmp,float* xi_sum, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_update_xisumKernel_init(const int printStats);
extern _CPPSTRING_ void EM_gammaKernel(const int N,const int current,size_t sm_size,const float* alpha,const float* beta,float* gamma, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_gammaKernel_init(const int printStats);
extern _CPPSTRING_ void EM_expectAKernel(const int N,size_t sm_size,const float* xi_sum,float* expect_A, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_expectAKernel_init(const int printStats);
extern _CPPSTRING_ void EM_gamma_state_sumKernel(const int N,const int T,size_t sm_size,const float* gamma,float* gamma_state_sum, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_gamma_state_sumKernel_init(const int printStats);
extern _CPPSTRING_ void EM_gamma_obsKernel(const int D,const int T,__constant float* constMem,const float* observations,float* gamma_obs, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_gamma_obsKernel_init(const int printStats);
extern _CPPSTRING_ void EM_expect_muKernel(const int D,const int T,const int offset,const int hs,size_t sm_size,const float* gamma_obs,const float* gamma_state_sum,float* expect_mu, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_expect_muKernel_init(const int printStats);
extern _CPPSTRING_ void EM_sigma_devKernel(const int D,const int T,const int hs,__constant float* constMem,const float* gamma_obs,const float* observations,const float* gamma_state_sum,float* sigma_dev, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_sigma_devKernel_init(const int printStats);
extern _CPPSTRING_ void EM_expect_sigmaKernel(const int blk_rows,const int width,const int start,const float* sigma_dev,float* expect_sigma, const snk_lparm_t * lparm);
extern _CPPSTRING_ void EM_expect_sigmaKernel_init(const int printStats);