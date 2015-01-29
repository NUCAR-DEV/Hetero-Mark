#include <iostream>
#include <string.h>
#include <clUtil.h>
#include "hmm.h"

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
        InitCL();
        InitKernels();
        InitBuffers();
}

void HMM::InitCL()
{
        // Init OCL context
        runtime    = clRuntime::getInstance();

        // OpenCL objects get from clRuntime class release themselves automatically, 
        // no need to clRelease them explicitly
        platform   = runtime->getPlatformID();
        device     = runtime->getDevice();
        context    = runtime->getContext();

        cmdQueue_0 = runtime->getCmdQueue(0);
        cmdQueue_1 = runtime->getCmdQueue(1);

        // Helper to read kernel file
        file       = clFile::getInstance();
}

void HMM::InitParam()
{
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
                dd             = D * D;

                tileblks       = (N/TILE) * (N/TILE);// [N/16][N/16]
                bytes_tileblks = sizeof(float) * tileblks;
        }
        else
        {
                std::cout << "Invalid N" << std::endl;
                exit(-1);
        }
}

void HMM::InitKernels()
{
        cl_int err;
        
        file->open("hmm_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();

        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source...\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");

        // Program build info
        // char buf[0x10000];
        // clGetProgramBuildInfo( program,
        //                         device,
        //                         CL_PROGRAM_BUILD_LOG,
        //                         0x10000,
        //                         buf,
        //                         NULL);
        // printf("\n%s\n", buf);

        // Create kernels
        kernel_FWD_init_alpha = clCreateKernel(program, "FWD_init_alpha", &err);
        checkOpenCLErrors(err, "Failed to create kernel FWD_init_alpha")
        kernel_FWD_scaling = clCreateKernel(program, "FWD_scaling", &err);
        checkOpenCLErrors(err, "Failed to create kernel FWD_scaling")
        kernel_FWD_calc_alpha = clCreateKernel(program, "FWD_calc_alpha", &err);
        checkOpenCLErrors(err, "Failed to create kernel FWD_calc_alpha")
        kernel_FWD_sum_ll = clCreateKernel(program, "FWD_sum_ll", &err);
        checkOpenCLErrors(err, "Failed to create kernel FWD_sum_ll")
        kernel_BK_update_beta = clCreateKernel(program, "BK_update_beta", &err);
        checkOpenCLErrors(err, "Failed to create kernel BK_update_beta")
        kernel_BK_scaling = clCreateKernel(program, "BK_scaling", &err);
        checkOpenCLErrors(err, "Failed to create kernel BK_scaling")
        kernel_EM_betaB_alphabeta = clCreateKernel(program, "EM_betaB_alphabeta", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_betaB_alphabeta")
        kernel_EM_alphabeta_update_gamma = clCreateKernel(program, "EM_alphabeta_update_gamma", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_alphabeta_update_gamma")
        kernel_EM_A_mul_alphabetaB = clCreateKernel(program, "EM_A_mul_alphabetaB", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_A_mul_alphabetaB")
        kernel_EM_update_xisum = clCreateKernel(program, "EM_update_xisum", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_update_xisum")
        kernel_EM_alphabeta = clCreateKernel(program, "EM_alphabeta", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_alphabeta")
        kernel_EM_expect_A = clCreateKernel(program, "EM_expect_A", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expect_A")
        kernel_EM_transpose = clCreateKernel(program, "EM_transpose", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_transpose")
        kernel_EM_gammastatesum = clCreateKernel(program, "EM_gammastatesum", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_gammastatesum")
        kernel_EM_gammaobs = clCreateKernel(program, "EM_gammaobs", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_gammaobs")
        kernel_EM_expectmu = clCreateKernel(program, "EM_expectmu", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expectmu")
        kernel_EM_expectsigma_dev = clCreateKernel(program, "EM_expectsigma_dev", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expectsigma_dev")
        kernel_EM_update_expectsigma = clCreateKernel(program, "EM_update_expectsigma", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_update_expectsigma")

}


void HMM::InitBuffers()
{

        // CPU buffers
        //-------------------------------------------------------------------------------------------//
        // SVM buffers 
        //        a,b,pi,lll, blk_result
        //-------------------------------------------------------------------------------------------//

        cl_int err;
        int i, j;

        bool svmCoarseGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_COARSE);
        bool svmFineGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_FINE);

        // Need at least coarse grain
        if (!svmCoarseGrainAvail)
        {
                printf("SVM coarse grain support unavailable\n");
                exit(-1);
        }

        // Alloc buffer
        if (!svmFineGrainAvail)
        {
                printf("SVM fine grain support unavailable\n");
               // state transition probability matrix
                a            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
                // emission probability matrix 
                b            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
                // prior probability
                pi           = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
                // intermediate blk results from the device
                blk_result   = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_tileblks, 0);
                // log likelihood 
                lll          = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float), 0);
                // forward probability matrix; hint: for checking purpose
                alpha        = (float *)malloc(bytes_nt);  // T x N
                // for em
                observations = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dt, 0);
        }
        else
        {
                printf("SVM fine grain support available\n");
               // state transition probability matrix
                a            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, bytes_nn, 0);
                // emission probability matrix 
                b            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, bytes_nt, 0);
                // prior probability
                pi           = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, bytes_n, 0);
                // intermediate blk results from the device
                blk_result   = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, bytes_tileblks, 0);
                // log likelihood 
                lll          = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, sizeof(float), 0);
                // forward probability matrix; hint: for checking purpose
                alpha        = (float *)malloc(bytes_nt);  // T x N
                // for em
                observations = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, bytes_dt, 0);               
        }

        // Sanity check
        if (!a || !b || !pi || !blk_result || !lll || !alpha || !observations)
        {
                printf("Cannot allocate SVM memory with clSVMAlloc\n");
                exit(-1);
        }

        // Coarse grain SVM needs explicit map/unmap
        if (!svmFineGrainAvail)
        {
               // Map a
                err = clEnqueueSVMMap(cmdQueue_0,
                                      CL_TRUE,       // blocking map
                                      CL_MAP_WRITE,
                                      a,
                                      bytes_nn,
                                      0, 0, 0
                                      );
                checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");
                // Map b
                err = clEnqueueSVMMap(cmdQueue_0,
                                      CL_TRUE,       // blocking map
                                      CL_MAP_WRITE,
                                      b,
                                      bytes_nt,
                                      0, 0, 0
                                      );
                checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");
                // Map pi
                err = clEnqueueSVMMap(cmdQueue_0,
                                      CL_TRUE,       // blocking map
                                      CL_MAP_WRITE,
                                      pi,
                                      bytes_n,
                                      0, 0, 0
                                      );
                checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");
                // Map observations
                err = clEnqueueSVMMap(cmdQueue_0,
                                      CL_TRUE,       // blocking map
                                      CL_MAP_WRITE,
                                      observations,
                                      bytes_dt,
                                      0, 0, 0
                                      );
                checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

        }

        // Init content
        for (i = 0; i < (N * N); i++)
                a[i] = 1.0f/(float)N;
        for (i = 0; i < (N * T); i++)
                b[i] = 1.0f/(float)T;
        for (i = 0; i < N; i++)
                pi[i] = 1.0f/(float)N;
        for(i = 0 ; i< T ; ++i)
                for(j = 0 ; j< D ; ++j)
                        observations[i * D + j] = (float)i + 1.f;

        // Coarse grain needs explicit unmap
        if (!svmFineGrainAvail)
        {
                err = clEnqueueSVMUnmap(cmdQueue_0, a, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
                err = clEnqueueSVMUnmap(cmdQueue_0, b, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
                err = clEnqueueSVMUnmap(cmdQueue_0, pi, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
                err = clEnqueueSVMUnmap(cmdQueue_0, observations, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
        }

        // GPU buffers
        // forward 
        ones_d              = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);      // for cublasdot
        ll_d                = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float)*(T + 1), 0);

        // backward
        beta_d              = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
        betaB_d             = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);

        // EM
        xi_sum_d            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
        alpha_beta_d        = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
        gamma_d             = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
        A_alphabetaB_d      = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
        gammaT_d            = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
        gamma_state_sum_d   = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
        gamma_obs_d         = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dt, 0);

        expect_prior_d      = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
        expect_A_d          = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);
        observationsT_d     = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dt, 0);

        expect_mu_d         = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dn, 0);
        expect_sigma_sym_d  = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dd, 0);
        expect_sigma_d      = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_ddn, 0);

}

void HMM::CleanUp()
{
        CleanUpKernels();
        CleanUpBuffers();
}


#define safeSVMFree(ctx, ptr) \
        if(ptr) \
          clSVMFree(ctx, ptr); 
void HMM::CleanUpBuffers()
{
        // CPU buffers
        if (alpha)
                free(alpha);
        safeSVMFree(context, a);
        safeSVMFree(context, b);
        safeSVMFree(context, pi);
        safeSVMFree(context, blk_result);
        safeSVMFree(context, observations);

        // GPU buffers
        // forward 
        safeSVMFree(context, ones_d);
        safeSVMFree(context, ll_d);

        // backward
        safeSVMFree(context, beta_d);
        safeSVMFree(context, betaB_d);

        // EM
        safeSVMFree(context, xi_sum_d);
        safeSVMFree(context, alpha_beta_d);
        safeSVMFree(context, gamma_d);
        safeSVMFree(context, A_alphabetaB_d);
        safeSVMFree(context, gammaT_d);
        safeSVMFree(context, gamma_state_sum_d);
        safeSVMFree(context, gamma_obs_d);

        safeSVMFree(context, expect_prior_d);
        safeSVMFree(context, expect_A_d);
        safeSVMFree(context, observationsT_d);

        safeSVMFree(context, expect_mu_d);
        safeSVMFree(context, expect_sigma_sym_d);
        safeSVMFree(context, expect_sigma_d);

}
#undef safeSVMFree

void HMM::CleanUpKernels()
{
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_init_alpha), "Failed to release kernel kernel_FWD_init_alpha");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_scaling), "Failed to release kernel kernel_FWD_scaling");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_calc_alpha), "Failed to release kernel kernel_FWD_calc_alpha");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_sum_ll), "Failed to release kernel kernel_FWD_sum_ll");
        checkOpenCLErrors(clReleaseKernel(kernel_BK_update_beta), "Failed to release kernel kernel_BK_update_beta");
        checkOpenCLErrors(clReleaseKernel(kernel_BK_scaling), "Failed to release kernel kernel_BK_scaling");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_betaB_alphabeta), "Failed to release kernel kernel_EM_betaB_alphabeta");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_alphabeta_update_gamma), "Failed to release kernel kernel_EM_alphabeta_update_gamma");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_A_mul_alphabetaB), "Failed to release kernel kernel_EM_A_mul_alphabetaB");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_update_xisum), "Failed to release kernel kernel_EM_update_xisum");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_alphabeta), "Failed to release kernel kernel_EM_alphabeta");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_expect_A), "Failed to release kernel kernel_EM_expect_A");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_transpose), "Failed to release kernel kernel_EM_transpose");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_gammastatesum), "Failed to release kernel kernel_EM_gammastatesum");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_gammaobs), "Failed to release kernel kernel_EM_gammaobs");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_expectmu), "Failed to release kernel kernel_EM_expectmu");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_expectsigma_dev), "Failed to release kernel kernel_EM_expectsigma_dev");
        checkOpenCLErrors(clReleaseKernel(kernel_EM_update_expectsigma), "Failed to release kernel kernel_EM_update_expectsigma");        
}

void HMM::Forward()
{
        ForwardInitAlpha(N, b, pi, alpha, ones_d, beta_d);

        ForwardSumAlpha();

        ForwardScaling(N, ll_d, 0, &alpha[0]);

        int frm;
        int current, previous;

        for (frm = 1; frm < T; ++frm) 
        {
            current  = frm * N; 
            previous = current - N;

            // a' * alpha
            // auto transposed due to the column major thing
            // ret = cublasSgemv(handle1, CUBLAS_OP_N, 
            //         N, N,
            //         &alp, 
            //         a_d, N, 
            //         &alpha[previous], 1,
            //         &bet, 
            //         &alpha[current], 1);

            // if (ret != CUBLAS_STATUS_SUCCESS) 
            // {
            //     fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
            //     exit(EXIT_FAILURE);
            // }

            // b * (a' * alpha) 
            ForwardCalcAlpha(N, &alpha[current] , &b[current]);

            // // the likelihood for current window
            // ret = cublasSdot(handle, N, 
            //         &alpha[current], 1, 
            //         ones_d, 1, 
            //         &ll_d[frm]);

            // if (ret != CUBLAS_STATUS_SUCCESS) 
            // {
            //     fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
            //     exit(EXIT_FAILURE);
            // }

            ForwardScaling(N, ll_d, frm, &alpha[current]);
        }

}

void HMM::ForwardInitAlpha(int numElements, float *bSrc, float *piSrc, float *alphaDst, float *onesDst, float *betaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_FWD_init_alpha, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 1, bSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 2, piSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 3, alphaDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 4, onesDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_init_alpha, 5, betaDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_FWD_init_alpha,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

}

void HMM::ForwardSumAlpha()
{
        // TODO 
}

void HMM::ForwardScaling(int numElements, float *scaleArraySrc, int scaleArrayIndexSrc, float *dataDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;
        int zero = 0;

        err = clSetKernelArg(kernel_FWD_scaling, 0, sizeof(int), (void*)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_scaling, 1, scaleArraySrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArg(kernel_FWD_scaling, 2, sizeof(int), &scaleArrayIndexSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_scaling, 3, dataDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_FWD_scaling,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

}

void HMM::ForwardCalcAlpha(int numElements, float *dst, float *src)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_FWD_calc_alpha, 0, sizeof(int), (void*)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_calc_alpha, 1, dst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_calc_alpha, 2, src);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_FWD_calc_alpha,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

}

void HMM::Backward()
{
    // beta_d is pre-computed in forward step

    int j;
    int current, previous;

    // Calcuate backwards 
    for(j = T-2; j >= 0; --j)
    {
        current = j * N;
        previous =  current + N;

        // betaB = beta(t) * b
        BackwardUpdateBeta(N, &beta_d[previous], &b[previous], betaB_d);

        // beta(t-1) = a * betaB
        // ret = cublasSgemv(handle1, CUBLAS_OP_T, 
        //         N, N, 
        //         &alp,
        //         a_d, N, 
        //         betaB_d, 1, 
        //         &bet, 
        //         &beta_d[current], 1);

        // if (ret != CUBLAS_STATUS_SUCCESS) 
        // {
        //     fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
        //     exit(EXIT_FAILURE);
        // }

        // sum up
        // ret = cublasSdot(handle, N, 
        //         &beta_d[current], 1, 
        //         ones_d, 1, 
        //         &ll_d[0]); // use ll_d[0] to save the sum

        // if (ret != CUBLAS_STATUS_SUCCESS) 
        // {
        //     fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
        //     exit(EXIT_FAILURE);
        // }

        // normalise
        BackwardScaling(N, &beta_d[current], ll_d);
    }

}

void HMM::BackwardUpdateBeta(int numElements, float *betaSrc, float *bSrc, float *betaBDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_BK_update_beta, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 1, betaSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 2, bSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 3, betaBDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_BK_update_beta,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::BackwardScaling(int numElements, float *llSrc, float *betaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_BK_update_beta, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 1, llSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_BK_update_beta, 2, betaDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_BK_update_beta,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");    

}

void HMM::BaumWelch()
{
    // clear the data for xi_sum
    clMemSet(cmdQueue_0, xi_sum_d, 0, bytes_nn);

    cl_int err;
    float sum;
    int window, i;
    int current, previous;
    uint start;

    for(window = 0; window < (T-1); ++window)
    {
        current = window * N;
        previous = current + N;

        // Calculate beta * B and alpha * beta
        EMBetaBAlphaBeta(N, current, previous, beta_d, b, alpha, betaB_d, alpha_beta_d);

        // ret = cublasSdot(handle, N,
        //         alpha_beta_d, 1, 
        //         ones_d, 1,
        //         &ll_d[0]);

        // if (ret != CUBLAS_STATUS_SUCCESS) 
        // {
        //     fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
        //     exit(EXIT_FAILURE);
        // }

        // Update gamma
        EMAlphaBetaUpdateGamma(N, current, alpha_beta_d, ll_d, gamma_d);

        // A .*  (alpha * betaB')
        EMAMulAlphaBetaB(N, a, A_alphabetaB_d, blk_result, &alpha[current], betaB_d);

        // Sum blkResult
        EMSumBlkresult(&sum);

        // Normalise A_alphabetaB and add up to xi_sum 
        EMUpdateXisum(N, sum, A_alphabetaB_d, xi_sum_d);

    }


}

void HMM::EMBetaBAlphaBeta(int numElements, int curWindow, int preWindow, 
        float *betaSrc, float *BSrc, float *alphaSrc, float *betaBDst, float *alphaBetaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_EM_betaB_alphabeta, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArg(kernel_EM_betaB_alphabeta, 1, sizeof(int), (void *)&curWindow);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArg(kernel_EM_betaB_alphabeta, 2, sizeof(int), (void *)&preWindow);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 3, betaSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 4, BSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 5, alphaSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 6, betaBDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_betaB_alphabeta, 7, alphaBetaDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_EM_betaB_alphabeta,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EMAlphaBetaUpdateGamma(int numElements, int curWindow, float *alphaBetaSrc,
        float *llSrc, float *gammaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = N / BLOCKSIZE;

        err = clSetKernelArg(kernel_EM_alphabeta_update_gamma, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArg(kernel_EM_alphabeta_update_gamma, 1, sizeof(int), (void *)&curWindow);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 2, alphaBetaSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 3, llSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 4, gammaDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_EM_alphabeta_update_gamma,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::EMAMulAlphaBetaB(int numElements, float *ASrc, float *AAlphaBetaBDst, 
        float *blkResultDst, float *constA, float *constB)
{
        cl_int err;

        size_t globalSize[2];
        size_t localSize[2];

        globalSize[0] = N;
        globalSize[1] = N;

        localSize[0] = N / 16;
        localSize[1] = N / 16;

        err = clSetKernelArg(kernel_EM_A_mul_alphabetaB, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 1, ASrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 2, AAlphaBetaBDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 3, blkResultDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArg(kernel_EM_A_mul_alphabetaB, 4, bytes_n, (void *)&constA);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArg(kernel_EM_A_mul_alphabetaB, 5, bytes_n, (void *)&constB);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_EM_A_mul_alphabetaB,
                2,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

// TODO: move to GPU
void HMM::EMSumBlkresult(float *sum)
{
        cl_int err;

        // Map
        err = clEnqueueSVMMap(cmdQueue_0,
                              CL_TRUE,       // blocking map
                              CL_MAP_WRITE,
                              blk_result,
                              bytes_tileblks,
                              0, 0, 0
                              );
        checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

        int i;
#pragma unroll
        for(i = 0; i < tileblks; ++i)
            *sum += blk_result[i];

        // Unmap
        err = clEnqueueSVMUnmap(cmdQueue_0, blk_result, 0, 0, 0);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
}

void HMM::EMUpdateXisum(int numElements, float sum, float *AAlphaBetaBSrc, float *xiSumDst)
{
        cl_int err;

        size_t globalSize[2];
        size_t localSize[2];

        globalSize[0] = N;
        globalSize[1] = N;

        localSize[0] = N / 16;
        localSize[1] = N / 16;

        err = clSetKernelArg(kernel_EM_update_xisum, 0, sizeof(int), (void *)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArg(kernel_EM_update_xisum, 1, sizeof(float), (void *)&sum);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_EM_update_xisum, 2, AAlphaBetaBSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_EM_update_xisum, 3, xiSumDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_EM_update_xisum,
                2,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");    
}

void HMM::Run()
{

        //-------------------------------------------------------------------------------------------//
        // HMM Parameters
        //        a,b,pi,alpha
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
        // Baum-Welch Algorithm on GPU 
        //-------------------------------------------------------------------------------------------//
        printf("\n");
        printf("      >> Start  Baum-Welch Algorithm on GPU.\n");
        BaumWelch();
        printf("      >> Finish Baum-Welch Algorithm on GPU.\n");

        printf("<=End program.\n");

}

int main(int argc, char const *argv[])
{
        if(argc != 2){
                puts("Please specify the number of hidden states N. (e.g., $./gpuhmmsr N)\nExit Program!");
                exit(1);
        }

        printf("=>Start program.\n");

        int N = atoi(argv[1]);

        // Smart pointer, auto cleanup in destructor
        std::unique_ptr<HMM> hmm(new HMM(N));

        hmm->Run();

        return 0;
}
