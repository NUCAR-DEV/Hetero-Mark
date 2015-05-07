#include "pagerank.h"

#include <memory>

PageRank::PageRank()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);
}

PageRank::~PageRank()
{
    FreeKernel();
    FreeBuffer();
}

void PageRank::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("pagerank_Kernels.cl");

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

void PageRank::InitBuffer()
{

}

void PageRank::FreeKernel()
{
        
}

void PageRank::FreeBuffer()
{
        
}

void PageRank::Run()
{
	//Input adjacency matrix from file

	//Use a kernel to convert the adajcency matrix to column stocastic matrix

	//The pagerank kernel where SPMV is iteratively called

}

int main(int argc, char const *argv[])
{
        std::unique_ptr<PageRank> pr(new PageRank());
    
        pr->Run();

        return 0;
}
