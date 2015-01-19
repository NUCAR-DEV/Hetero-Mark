#include "clUtil.h"

// Singleton instance
std::unique_ptr<clUtil> clUtil::instance;

clUtil *clUtil::getInstance()
{
	// Instance already exists
	if (instance.get())
		return instance.get();
	
	// Create instance
	instance.reset(new clUtil());
	return instance.get();
}

clUtil::clUtil()
{

}

clUtil::~clUtil()
{
	if (isInit)
	{
		clReleaseCommandQueue(cmdQueue);
		clReleaseContext(context);
	}

}

int clUtil::clInit()
{
	cl_int err = 0;
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	// Create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);

	// Create a command queue 
	cmdQueue = clCreateCommandQueue(context, device, 0, &err);

	// Mark as initialized
	isInit = true;

	return 0;
}

