#ifndef HSA_BENCHMARK_SHALLOW_WATER_H
#define HSA_BENCHMARK_SHALLOW_WATER_H

#include <clFile.h>

using namespace clHelper;

class shallowWater
{

public:
	shallowWater();
	~shallowWater();

	// Setup CL
	int setupCL();

	// Run CL
	int runCL();

	
};


#endif