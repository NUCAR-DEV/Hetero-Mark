#include "iir.h"

#include <memory>

IIR::IIR()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);
}

IIR::~IIR()
{
	FreeKernel();
	FreeBuffer();
}

void IIR::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("iir_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}

void IIR::InitBuffer()
{

}

void IIR::FreeKernel()
{
	
}

void IIR::FreeBuffer()
{
	
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<IIR> tp(new IIR());
	
	return 0;
}