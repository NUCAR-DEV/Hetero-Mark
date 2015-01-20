#include <iostream>

#include <clUtil.h>
#include <clFile.h>

#include "shallowWater.h"

using namespace clHelper;

shallowWater::shallowWater()
{

}

shallowWater::~shallowWater()
{

}

int shallowWater::setupCL()
{
	// Initialize OpenCL context/cmdQueue, etc.
	clUtil *cl_util = clUtil::getInstance();

	clFile *cl_file = clFile::getInstance();

	// Read kernel source code
	cl_file->open("shallowWater_Kernels.cl");

	return 0;
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<shallowWater> sw(new shallowWater);
	
	sw->setupCL();

	return 0;
}