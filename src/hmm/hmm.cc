#include <iostream>
#include <string.h>
#include <clUtil.h>
#include <math.h>
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
                bytes_const    = sizeof(float) * 4096;
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

        // EM
        kernel_EM_betaB_alphabeta = clCreateKernel(program, "EM_betaB_alphabeta", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_betaB_alphabeta")

        kernel_EM_sum_alphabeta = clCreateKernel(program, "EM_sum_alphabeta", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_sum_alphabeta")

        kernel_EM_alphabeta_update_gamma = clCreateKernel(program, "EM_alphabeta_update_gamma", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_alphabeta_update_gamma")

        kernel_EM_A_mul_alphabetaB = clCreateKernel(program, "EM_A_mul_alphabetaB", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_A_mul_alphabetaB")

        kernel_EM_update_xisum = clCreateKernel(program, "EM_update_xisum", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_update_xisum")

        kernel_EM_norm_alphabeta = clCreateKernel(program, "EM_norm_alphabeta", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_norm_alphabeta")

        kernel_EM_expt_A = clCreateKernel(program, "EM_expect_A", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expect_A")

        kernel_EM_transpose = clCreateKernel(program, "EM_transpose", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_transpose")

        kernel_EM_gammastatesum = clCreateKernel(program, "EM_gammastatesum", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_gammastatesum")

        kernel_EM_gammaobs = clCreateKernel(program, "EM_gammaobs", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_gammaobs")

        kernel_EM_exptmu = clCreateKernel(program, "EM_expectmu", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expectmu")

        kernel_EM_exptsigma_dev = clCreateKernel(program, "EM_expectsigma_dev", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_expectsigma_dev")

        kernel_EM_update_exptsigma = clCreateKernel(program, "EM_update_expectsigma", &err);
        checkOpenCLErrors(err, "Failed to create kernel EM_update_expectsigma")

}


void HMM::InitBuffers()
{

        // CPU buffers
        //-------------------------------------------------------------------------------------------//
        // SVM buffers 
        //        a,b,prior,lll, blk_result
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
                a = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

                // emission probability matrix 
                b = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);

                // prior probability
                prior = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);

                // intermediate blk results from the device
                blk_result = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_tileblks, 0);

                // log likelihood 
                lll = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float), 0);

                // forward probability matrix: TxN
                alpha = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);

                // observed input 
                observations = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_dt, 0);

                // Backward parameters 
                beta = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
                betaB = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);

                // EM parameters 
                alpha_beta = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_n, 0);
                gamma = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nt, 0);
                ll = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float) * (T + 1), 0);
                    
                // block results
                blk_result = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_tileblks, 0);

                A_alphabetaB = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

                xi_sum = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes_nn, 0);

                // constant memory buffers
                constA            = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_const, 0);
                constB            = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_const, 0);
                gamma_state_sumC  = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_const, 0);
                constT            = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_d, 0);
                expt_mu_state   = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, bytes_d, 0);
        }
        else
        {
                printf("SVM fine grain support available\n");

                // state transition probability matrix
                a = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_nn, 0);

                // emission probability matrix 
                b = (float *)clSVMAlloc(context,
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_nt, 0);

                // prior probability
                prior = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                                               bytes_n, 0);

                // intermediate blk results from the device
                blk_result = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_tileblks, 0);

                // log likelihood 
                lll = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               sizeof(float), 0);

                // forward probability matrix
                alpha = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_nt, 0);

                // for em
                observations = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_dt, 0);               

                // Backward parameters 
                beta = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_nt, 0);
                betaB = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_n, 0);

                // EM parameters 
                alpha_beta = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_n, 0);
                gamma = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_nt, 0);
                ll = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               sizeof(float) * (T + 1), 0);
                blk_result = (float *)clSVMAlloc(context, 
                                               CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 
                                               bytes_tileblks, 0);

                constA = (float *)clSVMAlloc(context, 
                                                CL_MEM_READ_ONLY | CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                                                bytes_const, 0);
                constB = (float *)clSVMAlloc(context, 
                                                CL_MEM_READ_ONLY | CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                                                bytes_const, 0);

                A_alphabetaB = (float *)clSVMAlloc(context, 
                                                CL_MEM_READ_ONLY | CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                                                bytes_nn, 0);

                xi_sum = (float *)clSVMAlloc(context, 
                                                CL_MEM_READ_ONLY | CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                                                bytes_nn, 0);
        }

        // Sanity check
        if (!a || !b || !prior || !blk_result || !lll || !alpha || !observations)
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
                // Map prior
                err = clEnqueueSVMMap(cmdQueue_0,
                                      CL_TRUE,       // blocking map
                                      CL_MAP_WRITE,
                                      prior,
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
                prior[i] = 1.0f/(float)N;
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
                err = clEnqueueSVMUnmap(cmdQueue_0, prior, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
                err = clEnqueueSVMUnmap(cmdQueue_0, observations, 0, 0, 0);
                checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");
        }
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
        safeSVMFree(context, prior);
        safeSVMFree(context, blk_result);
        safeSVMFree(context, observations);

        // GPU buffers
        // forward 
        safeSVMFree(context, ones);
        safeSVMFree(context, ll);

        // backward
        safeSVMFree(context, beta);
        safeSVMFree(context, betaB);

        // EM
        safeSVMFree(context, xi_sum);
        safeSVMFree(context, alpha_beta);
        safeSVMFree(context, gamma);
        safeSVMFree(context, A_alphabetaB);
        safeSVMFree(context, gammaT);
        safeSVMFree(context, gamma_state_sum);
        safeSVMFree(context, gamma_obs);

        safeSVMFree(context, expt_prior);
        safeSVMFree(context, expt_A);
        safeSVMFree(context, observationsT);

        safeSVMFree(context, expt_mu);
        safeSVMFree(context, expt_sigma_sym);
        safeSVMFree(context, expt_sigma);
}
#undef safeSVMFree

void HMM::CleanUpKernels()
{
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_init_alpha),
                          "Failed to release kernel kernel_FWD_init_alpha");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_scaling), 
                          "Failed to release kernel kernel_FWD_scaling");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_calc_alpha), 
                          "Failed to release kernel kernel_FWD_calc_alpha");
        checkOpenCLErrors(clReleaseKernel(kernel_FWD_sum_ll), 
                          "Failed to release kernel kernel_FWD_sum_ll");

        checkOpenCLErrors(clReleaseKernel(kernel_BK_update_beta), 
                          "Failed to release kernel kernel_BK_update_beta");
        checkOpenCLErrors(clReleaseKernel(kernel_BK_scaling), 
                          "Failed to release kernel kernel_BK_scaling");
        // EM
        checkOpenCLErrors(clReleaseKernel(kernel_EM_betaB_alphabeta), 
                          "Failed to release kernel kernel_EM_betaB_alphabeta");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_sum_alphabeta), 
                          "Failed to release kernel kernel_EM_sum_alphabeta");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_alphabeta_update_gamma), 
                          "Failed to release kernel kernel_EM_alphabeta_update_gamma");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_A_mul_alphabetaB), 
                          "Failed to release kernel kernel_EM_A_mul_alphabetaB");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_update_xisum), 
                          "Failed to release kernel kernel_EM_update_xisum");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_expt_A), 
                          "Failed to release kernel kernel_EM_expt_A");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_transpose), 
                          "Failed to release kernel kernel_EM_transpose");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_gammastatesum), 
                          "Failed to release kernel kernel_EM_gammastatesum");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_gammaobs), 
                          "Failed to release kernel kernel_EM_gammaobs");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_exptmu), 
                          "Failed to release kernel kernel_EM_exptmu");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_exptsigma_dev), 
                          "Failed to release kernel kernel_EM_exptsigma_dev");

        checkOpenCLErrors(clReleaseKernel(kernel_EM_update_exptsigma), 
                          "Failed to release kernel kernel_EM_update_exptsigma");        
}

void HMM::Forward()
{
        ForwardInitAlpha(N, b, prior, alpha, ones, beta);

        ForwardSumAlpha();

        ForwardScaling(N, ll, 0, alpha);

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
            ForwardCalcAlpha(N, &b[current], &alpha[current]);

            // // the likelihood for current window
            // ret = cublasSdot(handle, N, 
            //         &alpha[current], 1, 
            //         ones, 1, 
            //         &ll[frm]);

            // if (ret != CUBLAS_STATUS_SUCCESS) 
            // {
            //     fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
            //     exit(EXIT_FAILURE);
            // }

            ForwardScaling(N, ll, frm, &alpha[current]);
        }

        ForwardSumLL(N, ll);

}

void HMM::ForwardInitAlpha(int numElements, float *bSrc, float *piSrc, float *alphaDst, float *onesDst, float *betaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = 256;

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
        size_t localSize = 256;

        err = clSetKernelArg(kernel_FWD_scaling, 0, sizeof(int), (void*)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_scaling, 1, scaleArraySrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArg(kernel_FWD_scaling, 2, sizeof(int), (void*)&scaleArrayIndexSrc);
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

void HMM::ForwardCalcAlpha(int numElements, float *bSrc, float *alphaDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = 256;

        err = clSetKernelArg(kernel_FWD_calc_alpha, 0, sizeof(int), (void*)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_calc_alpha, 1, bSrc);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");
        err = clSetKernelArgSVMPointer(kernel_FWD_calc_alpha, 2, alphaDst);
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

void HMM::ForwardSumLL(int numElements, float *llDst)
{
        cl_int err;

        size_t globalSize = T;
        size_t localSize = T;

        err = clSetKernelArg(kernel_FWD_sum_ll, 0, sizeof(int), (void*)&numElements);
        checkOpenCLErrors(err, "Failed at clSetKernelArg");
        err = clSetKernelArgSVMPointer(kernel_FWD_sum_ll, 1, llDst);
        checkOpenCLErrors(err, "Failed at clSetKernelArgSVMPointer");

        err = clEnqueueNDRangeKernel(
                cmdQueue_0,
                kernel_FWD_sum_ll,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void HMM::Backward()
{
    // beta is pre-computed in forward step

    int j;
    int current, previous;

    // Calcuate backwards 
    for(j = T-2; j >= 0; --j)
    {
        current = j * N;
        previous = current + N;

        // betaB = beta(t) * b
        BackwardUpdateBeta(N, &beta[previous], &b[previous], betaB);

        // beta(t-1) = a * betaB
        // ret = cublasSgemv(handle1, CUBLAS_OP_T, 
        //         N, N, 
        //         &alp,
        //         a_d, N, 
        //         betaB, 1, 
        //         &bet, 
        //         &beta[current], 1);

        // if (ret != CUBLAS_STATUS_SUCCESS) 
        // {
        //     fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
        //     exit(EXIT_FAILURE);
        // }

        // sum up
        // ret = cublasSdot(handle, N, 
        //         &beta[current], 1, 
        //         ones, 1, 
        //         &ll[0]); // use ll[0] to save the sum

        // if (ret != CUBLAS_STATUS_SUCCESS) 
        // {
        //     fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
        //     exit(EXIT_FAILURE);
        // }

        // normalise
        BackwardScaling(N, &beta[current], ll);
    }

}

void HMM::BackwardUpdateBeta(int numElements, float *betaSrc, float *bSrc, float *betaBDst)
{
        cl_int err;

        size_t globalSize = N;
        size_t localSize = 256;

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
        size_t localSize = 256;

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
        cl_int err;
        int current, previous;
        float sum;

        for(int window = 0; window < (T - 1); ++window)
        {
                current = window * N;   
                previous = current + N;

                // alpha_beta summation
                // launch 1 block to sum up N points
                err  = clSetKernelArgSVMPointer(kernel_EM_sum_alphabeta, 0, (void *)(alpha_beta));
                err     |= clSetKernelArgSVMPointer(kernel_EM_sum_alphabeta, 1, (void *)(ll));
                err     |= clSetKernelArg(kernel_EM_sum_alphabeta, 2, sizeof(int), &N);
                err     |= clSetKernelArg(kernel_EM_sum_alphabeta, 3, sizeof(float) * 256, NULL);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                size_t local_01[1]  = {256};
                size_t global_01[1] = {256};
                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_sum_alphabeta,
                                1,
                                NULL,
                                global_01,
                                local_01,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");


                // update gamma 
                err  = clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 0, (void*)(alpha_beta));
                err     |= clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 1, (void*)(gamma));
                err     |= clSetKernelArgSVMPointer(kernel_EM_alphabeta_update_gamma, 2, (void*)(ll));
                err     |= clSetKernelArg(kernel_EM_alphabeta_update_gamma, 3, sizeof(int), &N);
                err     |= clSetKernelArg(kernel_EM_alphabeta_update_gamma, 4, sizeof(int), &current);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                size_t global_work_size[1] = {(size_t)N};
                size_t local_work_size[1] = {(size_t)(ceil(N/(float)255))*256};

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_alphabeta_update_gamma,
                                1,
                                NULL,
                                global_work_size,
                                local_work_size,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");

                // Copy alpha and betaB to constant memory 
                // err = clEnqueueCopyBuffer(cmdQueue_0, alpha, constA, current, 0, bytes_n, 0, NULL, NULL);
                // err = clEnqueueCopyBuffer(cmdQueue_0, beta,  constB, 0,       0, bytes_n, 0, NULL, NULL);
                err  = clEnqueueSVMMemcpy(cmdQueue_0, true, constA, &alpha[current], bytes_n, 0, NULL, NULL);
                err |= clEnqueueSVMMemcpy(cmdQueue_0, true, constB, betaB, bytes_n, 0, NULL, NULL);
                checkOpenCLErrors(err, "Failed to copy alpha and betaB to constant memory");


                // A . * (alpha * betaB') 
                err  = clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 0, (void*)(a));
                err     |= clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 1, (void*)(A_alphabetaB));
                err     |= clSetKernelArgSVMPointer(kernel_EM_A_mul_alphabetaB, 2, (void*)(blk_result));
                err     |= clSetKernelArg(kernel_EM_A_mul_alphabetaB, 3, bytes_const, constA);
                err     |= clSetKernelArg(kernel_EM_A_mul_alphabetaB, 4, bytes_const, constB);
                err     |= clSetKernelArg(kernel_EM_A_mul_alphabetaB, 5, sizeof(int), &N);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                size_t local_2d_nn[2]  = {16,16};
                size_t global_2d_nn[2] = {(size_t)(ceil(N/(float)15)*16), (size_t)(ceil(N/(float)15)*16)};
                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_A_mul_alphabetaB,
                                2,
                                NULL,
                                global_2d_nn,
                                local_2d_nn,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");

                // Map and Unmap the buffer, to sum up the blk results on the CPU
                err = clEnqueueSVMMap(cmdQueue_0,
                                         CL_TRUE,
                                         CL_MAP_READ,
                                         blk_result,
                                         bytes_tileblks,
                                         0, NULL, NULL);
                checkOpenCLErrors(err, "Failed to map buffer!");

                sum = 0.f;
#pragma unroll
                for(int i = 0; i < tileblks; ++i){
                        sum += blk_result[i];
                }

                err = clEnqueueSVMUnmap(cmdQueue_0, blk_result, 0, NULL, NULL);
                checkOpenCLErrors(err, "Failed to unmap buffer!");


                // Normalise A_alphabetaB and add to xi_sum
                err  = clSetKernelArgSVMPointer(kernel_EM_update_xisum, 0, (void*)(A_alphabetaB));
                err     |= clSetKernelArgSVMPointer(kernel_EM_update_xisum, 1, (void*)(xi_sum));
                err     |= clSetKernelArg(kernel_EM_update_xisum, 2, sizeof(float), &sum);
                err     |= clSetKernelArg(kernel_EM_update_xisum, 3, sizeof(int),   &N);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_update_xisum,
                                2,
                                NULL,
                                global_2d_nn,
                                local_2d_nn,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");
        }

        current = previous;

        // Normalise (alpha .* beta) at the last window frame, update the gamma
        // Launch 1 workgroup
        err  = clSetKernelArgSVMPointer(kernel_EM_norm_alphabeta, 0, (void*)(alpha));
        err     |= clSetKernelArgSVMPointer(kernel_EM_norm_alphabeta, 1, (void*)(beta));
        err     |= clSetKernelArgSVMPointer(kernel_EM_norm_alphabeta, 2, (void*)(alpha_beta));
        err     |= clSetKernelArgSVMPointer(kernel_EM_norm_alphabeta, 3, (void*)(gamma));
        err     |= clSetKernelArg(kernel_EM_norm_alphabeta, 4, sizeof(float)*256, NULL);
        err     |= clSetKernelArg(kernel_EM_norm_alphabeta, 5, sizeof(int), &current);
        err     |= clSetKernelArg(kernel_EM_norm_alphabeta, 6, sizeof(int),   &N);
        checkOpenCLErrors(err, "Failed to configure kernel arguments!");

        // FIXME
        size_t global_01[2] = {(size_t)N, (size_t)N};
        size_t local_01[2] = {16, 16};

        err = clEnqueueNDRangeKernel(
                        cmdQueue_0,
                        kernel_EM_update_xisum,
                        2,
                        NULL,
                        global_01,
                        local_01,
                        0,
                        NULL,
                        NULL);
        checkOpenCLErrors(err, "Failed to execute kernel!");


        // Update expected prior prob, copy memory buffer 
        // expected_prior = gamma(:, 1);
        // err = clEnqueueCopyBuffer(cmdQueue_0, gamma, expt_prior, 0, 0, bytes_n, 0, NULL, NULL);
        err = clEnqueueSVMMemcpy(cmdQueue_0, true, expt_prior, gamma, bytes_n, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to copy gamma to expt_prior");

        // expected_A     = mk_stochastic(xi_sum);
        size_t local_2d_1n[2]  = {16,16};
        size_t global_2d_1n[2] = {16, (size_t)(ceil(N/(float)15)*16)};

        err  = clSetKernelArgSVMPointer(kernel_EM_expt_A, 0, (void*)(xi_sum));
        err     |= clSetKernelArgSVMPointer(kernel_EM_expt_A, 1, (void*)(expt_A));
        err     |= clSetKernelArg(kernel_EM_expt_A, 2, sizeof(int), &N);
        checkOpenCLErrors(err, "Failed to configure kernel arguments!");

        err = clEnqueueNDRangeKernel(
                        cmdQueue_0,
                        kernel_EM_expt_A,
                        2,
                        NULL,
                        global_2d_1n,
                        local_2d_1n,
                        0,
                        NULL,
                        NULL);
        checkOpenCLErrors(err, "Failed to execute kernel!");


        // Transpose gamma
        size_t local_2d_nt[2]  = {16,16};
        size_t global_2d_nt[2] = {(size_t)(ceil(N/(float)15)*16), (size_t)(ceil(T/(float)15)*16)};

        err  = clSetKernelArgSVMPointer(kernel_EM_transpose, 0, (void*)(gamma));
        err     |= clSetKernelArgSVMPointer(kernel_EM_transpose, 1, (void*)(gammaT));
        err     |= clSetKernelArg(kernel_EM_transpose, 2, sizeof(int), &T); // rows 
        err     |= clSetKernelArg(kernel_EM_transpose, 3, sizeof(int), &N); // columns 
        checkOpenCLErrors(err, "Failed to configure kernel arguments!");

        err = clEnqueueNDRangeKernel(
                        cmdQueue_0,
                        kernel_EM_transpose,
                        2,
                        NULL,
                        global_2d_nt,
                        local_2d_nt,
                        0,
                        NULL,
                        NULL);
        checkOpenCLErrors(err, "Failed to execute kernel!");


        // Compute gamma_state_sum: row sum of gamma (N x T)
        size_t local_2d_tn[2]  = {16,16};
        size_t global_2d_tn[2] = {(size_t)(ceil(T/(float)15)*16), (size_t)(ceil(N/(float)15)*16)};

        err  = clSetKernelArgSVMPointer(kernel_EM_gammastatesum, 0, (void*)(gammaT));
        err     |= clSetKernelArgSVMPointer(kernel_EM_gammastatesum, 1, (void*)(gamma_state_sum));
        err     |= clSetKernelArg(kernel_EM_gammastatesum, 2, sizeof(int), &N); // rows 
        err     |= clSetKernelArg(kernel_EM_gammastatesum, 3, sizeof(int), &T); // rows 
        checkOpenCLErrors(err, "Failed to configure kernel arguments!");

        err = clEnqueueNDRangeKernel(
                        cmdQueue_0,
                        kernel_EM_gammastatesum,
                        2,
                        NULL,
                        global_2d_tn,
                        local_2d_tn,
                        0,
                        NULL,
                        NULL);
        checkOpenCLErrors(err, "Failed to execute kernel!");

        // Copy gamma_state_sum to constant memory
        // err = clEnqueueCopyBuffer(cmdQueue_0, gamma_state_sum, gamma_state_sumC, 0, 0, bytes_n, 0, NULL, NULL);
        err = clEnqueueSVMMemcpy(cmdQueue_0, true, gamma_state_sumC, gamma_state_sum, bytes_n, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to copy gamma_state_sum to gamma_state_sumC");

        // Transpose observations
        size_t local_2d_dt[2]  = {16,16};
        size_t global_2d_dt[2] = {(size_t)(ceil(D/(float)15)*16), (size_t)(ceil(T/(float)15)*16)};

        err  = clSetKernelArgSVMPointer(kernel_EM_transpose, 0, (void*)(observations));
        err     |= clSetKernelArgSVMPointer(kernel_EM_transpose, 1, (void*)(observationsT));
        err     |= clSetKernelArg(kernel_EM_transpose, 2, sizeof(int), &T); // rows 
        err     |= clSetKernelArg(kernel_EM_transpose, 3, sizeof(int), &D); // columns 
        checkOpenCLErrors(err, "Failed to configure kernel arguments!");

        err = clEnqueueNDRangeKernel(
                        cmdQueue_0,
                        kernel_EM_transpose,
                        2,
                        NULL,
                        global_2d_dt,
                        local_2d_dt,
                        0,
                        NULL,
                        NULL);
        checkOpenCLErrors(err, "Failed to execute kernel!");


        // Update mean and variance for each hidden state
        int start;
        for(int hs = 0; hs < N; ++hs)
        {
                // Copy gammaT to constant mem  
                // FIXME: bufferT?
                // err = clEnqueueCopyBuffer(cmdQueue_0, gammaT, bufferT, hs*T, 0, bytes_t, 0, NULL, NULL);
                // err = clEnqueueSVMMemcpy(cmdQueue_0, true, bufferT[hs*T], gammaT, bytes_t, 0, NULL, NULL);
                // checkOpenCLErrors(err, "Failed to copy gammaT to bufferT[hs*T]");

                // Compute gamma_obs
                size_t local_2d_td[2]  = {16, 16};
                size_t global_2d_td[2] = {(size_t)(ceil(T/(float)15)*16), (size_t)(ceil(D/(float)15)*16)};

                err  = clSetKernelArgSVMPointer(kernel_EM_gammaobs, 0, (void*)(observationsT));
                err  = clSetKernelArgSVMPointer(kernel_EM_gammaobs, 1, (void*)(gamma_obs));
                err  = clSetKernelArgSVMPointer(kernel_EM_gammaobs, 2, (void*)(constT));
                err     |= clSetKernelArg(kernel_EM_gammaobs, 3, sizeof(int), &T); // rows 
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_gammaobs,
                                2,
                                NULL,
                                global_2d_td,
                                local_2d_td,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");

                current = hs * D;

                // Update Expected Mu
                size_t local_2d_1d[2]  = {16, 16};
                size_t global_2d_1d[2] = {16, (size_t)(ceil(D/(float)15)*16)};

                err  = clSetKernelArgSVMPointer(kernel_EM_exptmu, 0, (void*)(gamma_obs));
                err  = clSetKernelArgSVMPointer(kernel_EM_exptmu, 1, (void*)(expt_mu));
                err  = clSetKernelArgSVMPointer(kernel_EM_exptmu, 2, (void*)(gamma_state_sumC));
                err     |= clSetKernelArg(kernel_EM_exptmu, 3, sizeof(int), &hs);
                err     |= clSetKernelArg(kernel_EM_exptmu, 4, sizeof(int), &T);
                err     |= clSetKernelArg(kernel_EM_exptmu, 5, sizeof(int), &current);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_exptmu,
                                2,
                                NULL,
                                global_2d_td,
                                local_2d_td,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");


                // copy the current state row to constant memory
                // err = clEnqueueCopyBuffer(cmdQueue_0, expt_mu, expt_mu_state, hs*D, 0, bytes_d, 0, NULL, NULL);
                err = clEnqueueSVMMemcpy(cmdQueue_0, true, &expt_mu_state[hs*D], expt_mu, bytes_d, 0, NULL, NULL );
                checkOpenCLErrors(err, "Failed to copy current state row to constant memory");

                // Calculate the symmetric expt_sigma
                size_t local_2d_dd[2]  = {8, 8};
                size_t global_2d_dd[2] = {(size_t)(ceil(D/(float)7)*8), (size_t)(ceil(D/(float)7)*8)};

                err  = clSetKernelArgSVMPointer(kernel_EM_exptsigma_dev, 0, (void*)(gamma_obs));
                err |= clSetKernelArgSVMPointer(kernel_EM_exptsigma_dev, 1, (void*)(observations));
                err |= clSetKernelArgSVMPointer(kernel_EM_exptsigma_dev, 2, (void*)(expt_sigma_sym));
                err |= clSetKernelArgSVMPointer(kernel_EM_exptsigma_dev, 3, (void*)(gamma_state_sumC));
                err |= clSetKernelArgSVMPointer(kernel_EM_exptsigma_dev, 4, (void*)(expt_mu_state));
                err     |= clSetKernelArg(kernel_EM_exptsigma_dev, 5, sizeof(int), &hs);
                err     |= clSetKernelArg(kernel_EM_exptsigma_dev, 6, sizeof(int), &D);
                err     |= clSetKernelArg(kernel_EM_exptsigma_dev, 7, sizeof(int), &T);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_exptsigma_dev,
                                2,
                                NULL,
                                global_2d_dd,
                                local_2d_dd,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");

                // Update expt_sigma for each hidden state
                start = hs * D;

                size_t local_2d_blknum[2]  = {16, 16};
                size_t global_2d_blknum[2] = {16, (size_t)blknum};

                err  = clSetKernelArgSVMPointer(kernel_EM_update_exptsigma, 0, (void*)(expt_sigma));
                err |= clSetKernelArgSVMPointer(kernel_EM_update_exptsigma, 1, (void*)(expt_sigma_sym));
                err     |= clSetKernelArg(kernel_EM_update_exptsigma, 2, sizeof(int), &blk_rows);
                err     |= clSetKernelArg(kernel_EM_update_exptsigma, 3, sizeof(int), &D);
                err     |= clSetKernelArg(kernel_EM_update_exptsigma, 4, sizeof(int), &start);
                checkOpenCLErrors(err, "Failed to configure kernel arguments!");

                err = clEnqueueNDRangeKernel(
                                cmdQueue_0,
                                kernel_EM_update_exptsigma,
                                2,
                                NULL,
                                global_2d_blknum,
                                local_2d_blknum,
                                0,
                                NULL,
                                NULL);
                checkOpenCLErrors(err, "Failed to execute kernel!");
        }
}

void HMM::Run()
{

        //---------------------------------------------------------------------------------------//
        // HMM Parameters
        //        a,b,prior,alpha
        //---------------------------------------------------------------------------------------//
        printf("=>Initialize parameters.\n");
        Init();

        //---------------------------------------------------------------------------------------//
        // Forward Algorithm on GPU 
        //---------------------------------------------------------------------------------------//
        printf("\n");
        printf("      >> Start  Forward Algorithm on GPU.\n");
        Forward();
        printf("      >> Finish Forward Algorithm on GPU.\n");

        //---------------------------------------------------------------------------------------//
        // Backward Algorithm on GPU 
        //---------------------------------------------------------------------------------------//
        printf("\n");
        printf("      >> Start  Backward Algorithm on GPU.\n");
        Backward();
        printf("      >> Finish Backward Algorithm on GPU.\n");

        //---------------------------------------------------------------------------------------//
        // Baum-Welch Algorithm on GPU 
        //---------------------------------------------------------------------------------------//
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


