#include <CL/cl.h>
#include <memory>

#ifndef CL_UTIL_H
#define CL_UTIL_H 

class clUtil
{
	// Instance of the singleton
	static std::unique_ptr<clUtil> instance;

	// Private constructor for singleton
	clUtil();

	// OpenCL runtime
	bool isInit;
	cl_platform_id   platform;
 	cl_context       context;
	cl_device_id     device;
	cl_command_queue cmdQueue;

public:
	// Destructor
	~clUtil();

	static clUtil *getInstance();

	int clInit();

};


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



#endif
