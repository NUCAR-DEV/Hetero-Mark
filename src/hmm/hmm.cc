#include <iostream>
#include <string.h>
#include <clUtil.h>
#include "hmm.h"

HMM::HMM(int N)
{
        this->N = N;
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
        runtime = clRuntime::getInstance();

        // OpenCL objects get from clRuntime class release themselves automatically, 
        // no need to clRelease them explicitly
        platform = runtime->getPlatformID();
        device = runtime->getDevice();
        context = runtime->getContext();

        runtime->displayAllInfo();

        cmdQueue_0 = runtime->getCmdQueue(0);
        cmdQueue_1 = runtime->getCmdQueue(1);

        // Helper to read kernel file
        file = clFile::getInstance();
        file->open("hmm_Kernels.cl");
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
        cl_svm_mem_flags flags = CL_MEM_READ_WRITE;

        // Need at least coarse grain
        if (!svmCoarseGrainAvail)
        {
                printf("SVM coarse grain support unavailable\n");
                exit(-1);
        }

        // Fine grain
        if (!svmFineGrainAvail)
                printf("SVM fine grain support unavailable\n");
        else
                flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;

        // state transition probability matrix
        a            = (float *)clSVMAlloc(context, flags, bytes_nn, 0);
        // emission probability matrix 
        b            = (float *)clSVMAlloc(context, flags, bytes_nt, 0);
        // prior probability
        pi           = (float *)clSVMAlloc(context, flags, bytes_n, 0);
        // intermediate blk results from the device
        blk_result   = (float *)clSVMAlloc(context, flags, bytes_tileblks, 0);
        // log likelihood 
        lll          = (float *)clSVMAlloc(context, flags, sizeof(float), 0);
        // forward probability matrix; hint: for checking purpose
        alpha        = (float *)malloc(bytes_nt);  // T x N
        // for em
        observations = (float *)clSVMAlloc(context, flags, bytes_dt, 0);

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

        for (i = 0; i < (N * N); i++)
                a[i] = 1.0f/(float)N;
        for (i = 0; i < (N * T); i++)
                b[i] = 1.0f/(float)T;
        for (i = 0; i < N; i++)
                pi[i] = 1.0f/(float)N;
        for(i = 0 ; i< T ; ++i)
                for(j = 0 ; j< D ; ++j)
                        observations[i * D + j] = (float)i + 1.f;

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

}

void HMM::CleanUp()
{
        CleanUpKernels();
        CleanUpBuffers();
}

void HMM::CleanUpBuffers()
{

}

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

}

void HMM::Backward()
{

}

void HMM::BaumWelch()
{

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
        // Forward Algorithm on GPU 
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