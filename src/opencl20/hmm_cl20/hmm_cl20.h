#ifndef HMM_H
#define HMM_H

#include "src/common/cl_util/cl_util.h"
#include "src/common/benchmark/benchmark.h"

using namespace clHelper;

class HMM : public Benchmark {
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
  // cl_command_queue cmdQueue_1;

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
  const int T = 64;  // number of (overlapping) windows
  const int D = 64;  // number of features

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
  float *a;             // state transition probability matrix
  float *b;             // emission probability matrix
  float *alpha;         // forward probability matrix
  float *prior;         // prior probability
  float *observations;  // D x T

  // Forward
  float *lll;  // log likelihood
  float *aT;   // transpose of a

  // Backward
  float *beta;
  float *betaB;

  // EM
  float *xi_sum;       // N x N
  float *alpha_beta;   // N
  float *gamma;        // T x N
  float *alpha_betaB;  // N x N
  float *xi_sum_tmp;   // N x N
  float *blk_result;   // intermediate blk results

  float *expect_prior;  // N
  float *expect_A;      // N xN
  float *expect_mu;     // N x D
  float *expect_sigma;  // N x D x D

  float *gamma_state_sum;  // N
  float *gamma_obs;        // D x T
  float *sigma_dev;        // D x D

  // Constant
  float *constMem;

  //--------------------------------------------------------------------------//
  // Initialize functions
  //--------------------------------------------------------------------------//
  void Initialize() override { Init(); }
  void Init();
  void InitParam();
  void InitCL();
  void InitKernels();
  void InitBuffers();

  //--------------------------------------------------------------------------//
  // Clean functions
  //--------------------------------------------------------------------------//
  void Cleanup() override;
  void CleanUpKernels();
  void CleanUpBuffers();

  //--------------------------------------------------------------------------//
  // Forward functions
  //--------------------------------------------------------------------------//
  void Forward();
  void ForwardInitAlpha();
  void ForwardNormAlpha(int startpos);
  void TransposeSym(float *a, float *aT, int size);
  void ForwardUpdateAlpha(int pos);

  //--------------------------------------------------------------------------//
  // Backward functions
  //--------------------------------------------------------------------------//
  void Backward();
  void BackwardBetaB(int pos);
  void BackwardUpdateBeta(int pos);
  void BackwardNormBeta(int pos);

  //--------------------------------------------------------------------------//
  // EM functions
  //--------------------------------------------------------------------------//
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
  HMM();
  ~HMM();

  void SetInitialParameters(int num_states) {
    if (num_states >= TILE && ((num_states % TILE) == 0)) {
      this->N = num_states;
    } else {
      std::cout << "Invalid value for hidden_state_num = " << num_states
                << ".\n";
      std::cout << "The hidden_state_num should be greater than " << TILE;
      std::cout << ", and evenly divisible by " << TILE << ".\n";
      exit(-1);
    }
  }

  void Run() override;
  void Verify() override {}
  void Summarize() override {}
};

#endif
