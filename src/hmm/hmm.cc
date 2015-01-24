#include <iostream>

#include <clUtil.h>

#include "hmm.h"

HMM::HMM(int N)
{
	this->N = N;

	// Init OCL context
	runtime = clRuntime::getInstance();
	runtime->displayAllInfo();

	file = clFile::getInstance();

}

HMM::~HMM()
{
	// Some cleanup
	Release();
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

int main(int argc, char const *argv[])
{
	if(argc != 2){
		puts("Please specify the number of hidden states N. (e.g., $./gpuhmmsr N)\nExit Program!");
		exit(1);
	}

	printf("=>Start program.\n");

	int N = atoi(argv[1]);

	// Smart pointer, auto cleanup
	std::unique_ptr<HMM> hmm(new HMM(N));

	//-------------------------------------------------------------------------------------------//
	// HMM Parameters
	//	a,b,pi,alpha
	//-------------------------------------------------------------------------------------------//
	printf("=>Initialize parameters.\n");
	hmm->Param();

	//-------------------------------------------------------------------------------------------//
	// Forward Algorithm on GPU 
	//-------------------------------------------------------------------------------------------//
	printf("\n");
	printf("      >> Start  Forward Algorithm on GPU.\n");
	hmm->Forward();
	printf("      >> Finish Forward Algorithm on GPU.\n");

	//-------------------------------------------------------------------------------------------//
	// Forward Algorithm on GPU 
	//-------------------------------------------------------------------------------------------//
	printf("\n");
	printf("      >> Start  Backward Algorithm on GPU.\n");
	hmm->Backward();
	printf("      >> Finish Backward Algorithm on GPU.\n");

	//-------------------------------------------------------------------------------------------//
	// Baum-Welch Algorithm on GPU 
	//-------------------------------------------------------------------------------------------//
	printf("\n");
	printf("      >> Start  Baum-Welch Algorithm on GPU.\n");
	hmm->BaumWelch();
	printf("      >> Finish Baum-Welch Algorithm on GPU.\n");

	printf("<=End program.\n");

	return 0;
}