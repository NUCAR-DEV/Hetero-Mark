#include "aes.h"

#include <memory>

AES::AES()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);
}

AES::~AES()
{

}

void AES::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("aes_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        if (err != CL_SUCCESS)
        {
            char buf[0x10000];
            clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  0x10000,
                                  buf,
                                  NULL);
            printf("\n%s\n", buf);
            exit(-1);
        }

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}

void AES::InitBuffer()
{

}

void AES::FreeKernel()
{
        
}

void AES::FreeBuffer()
{
        
}

int main(int argc, char const *argv[])
{
        std::unique_ptr<AES> aes(new AES());
        
        return 0;
}