#include <iostream>

#include <clUtil.h>
#include <clFile.h>
#include <clError.h>

#include "shallowWater.h"

using namespace clHelper;

shallowWater::shallowWater()
{
	// Initialize OpenCL context/cmdQueue
	cl_util = clUtil::getInstance();

	// Helper class to read kernel source code
	cl_file = clFile::getInstance();

}

shallowWater::~shallowWater()
{

}

int shallowWater::setupCL()
{
	cl_int err;

	clFlush(cl_util->getCmdQueue());
	clFinish(cl_util->getCmdQueue());

	// Some cleanup
	if( kernel )
	{
		clReleaseKernel( kernel );
		kernel = 0;
	}

	if( program )
	{
		clReleaseProgram( program );
		program = 0;
	}

	// Read kernel source code
	cl_file->open("shallowWater_Kernels.cl");
	
	// Create program
	const char *source = cl_file->getSourceChar();
	program = clCreateProgramWithSource(cl_util->getContext(), 1, 
		(const char**)&source, NULL, &err);
	CHECK_OPENCL_ERROR(err, "ERROR: Failed to create Program with source...\n");

	// Create kernel with OpenCL 2.0 support
	err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	CHECK_OPENCL_ERROR(err, "ERROR: Failed to build program...\n");

	kernel = clCreateKernel(program, "ProcessTile", &err);
	CHECK_OPENCL_ERROR(err, "ERROR: Failed to create kernel...\n");

	// std::cout << cl_file->getSource() << std::endl;

	return 0;
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<shallowWater> sw(new shallowWater);
	
	sw->setupCL();

	return 0;
}