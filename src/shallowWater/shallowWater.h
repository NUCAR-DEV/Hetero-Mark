#ifndef HSA_BENCHMARK_SHALLOW_WATER_H
#define HSA_BENCHMARK_SHALLOW_WATER_H

#include <clFile.h>

using namespace clHelper;

class shallowWater
{
	clUtil *cl_util;
	clFile *cl_file;

	cl_program program;
	cl_kernel  kernel;

	size_t global_size[1];
	size_t local_size[1];

public:
	shallowWater();
	~shallowWater();

	// Setup CL
	int setupCL();

	// Run CL
	int runCL();

	
};


#endif