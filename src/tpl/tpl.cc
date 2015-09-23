 #include "tpl.h"

#include <memory>

TPL::TPL()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);
}

TPL::~TPL()
{
    FreeKernel();
    FreeBuffer();
}

void TPL::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("tpl_Kernels.cl");

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

void TPL::InitBuffer()
{

}

void TPL::FreeKernel()
{
        
}

void TPL::FreeBuffer()
{
        
}

void TPL::Run()
{

}

int main(int argc, char const *argv[])
{
        std::unique_ptr<TPL> tpl(new TPL());
    
        tpl->Run();

        return 0;
}