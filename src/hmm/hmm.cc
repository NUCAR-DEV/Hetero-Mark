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

	printf("%s\n", source);
	program = clCreateProgramWithSource(context, 1, 
		(const char**)&source, NULL, &err);
	checkOpenCLErrors(err, "Failed to create Program with source...\n");

	// Create kernel with OpenCL 2.0 support
	err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	checkOpenCLErrors(err, "Failed to build program...\n");

	char buf[0x10000];
	clGetProgramBuildInfo( program,
				device,
				CL_PROGRAM_BUILD_LOG,
				0x10000,
				buf,
				NULL);
	printf("\n%s\n", buf);

}

void HMM::Param()
{

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