#include <iostream>

#include <clUtil.h>

#include "hmm.h"

HMM::HMM(int N)
{
	this->N = N;
}

HMM::~HMM()
{
	// Some cleanup
	Release();
}

void HMM::SetupCL()
{
	// Init OCL context
	runtime = clRuntime::getInstance();
	device = runtime->getDevice();
	context = runtime->getContext();

	// Helper to read kernel file
	file = clFile::getInstance();
	file->open("hmm_Kernels.cl");

	cl_int err;
	
	// Create program
	const char *source = file->getSourceChar();

	program = clCreateProgramWithSource(context, 1, 
		(const char**)&source, NULL, &err);
	checkOpenCLErrors(err, "Failed to create Program with source...\n");

	// Create program with OpenCL 2.0 support
	err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	checkOpenCLErrors(err, "Failed to build program...\n");

	// Program build info
	char buf[0x10000];
	clGetProgramBuildInfo( program,
				device,
				CL_PROGRAM_BUILD_LOG,
				0x10000,
				buf,
				NULL);
	printf("\n%s\n", buf);

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

void HMM::Param()
{
	if (!N)
	{
		bytes_nn  = sizeof(float) * N * N;
		bytes_nt  = sizeof(float) * N * T;
		bytes_n   = sizeof(float) * N;
		bytes_dt  = sizeof(float) * D * T;
		bytes_dd  = sizeof(float) * D * D;
		bytes_dn  = sizeof(float) * D * N ;
		bytes_ddn = sizeof(float) * D * D * N ;
		bytes_t   = sizeof(float) * T;
		bytes_d   = sizeof(float) * D;
		bytes_n   = sizeof(float) * N;
		dd = D * D;

		tileblks = (N/TILE) * (N/TILE);// [N/16][N/16]
		bytes_tileblks = sizeof(float) * tileblks;
	}
	else
		std::cout << "Invalid N" << std::endl;

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

void HMM::Release()
{

}

void HMM::Run()
{

	SetupCL();

	//-------------------------------------------------------------------------------------------//
	// HMM Parameters
	//	a,b,pi,alpha
	//-------------------------------------------------------------------------------------------//
	printf("=>Initialize parameters.\n");
	Param();

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