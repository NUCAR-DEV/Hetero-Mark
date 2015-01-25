#ifndef CL_RUNTIME_H
#define CL_RUNTIME_H 

#include <memory>
#include <vector>
#include <CL/cl.h>

#include "clError.h"

namespace clHelper
{

// OpenCL runtime contains objects don't change during execution
// These objects are automatically freed at the end of execution
class clRuntime
{

private:

	cl_platform_id   platform;
	cl_device_id     device;
 	cl_context       context;
 	
 	std::vector<cl_command_queue> cmdQueueRepo;

	// Instance of the singleton
	static std::unique_ptr<clRuntime> instance;

	// Private constructor for singleton
	clRuntime();

	int displayPlatformInfo(cl_platform_id plt_id, 
		cl_platform_info plt_info);

	int displayContextInfo(cl_context ctx, 
		cl_context_info ctx_info);

public:
	// Destructor
	~clRuntime();

	// Get singleton
	static clRuntime *getInstance();

	/// Getters
	cl_platform_id const getPlatformID() { return platform; }

	cl_device_id const getDevice() { return device; }

	cl_context const getContext() { return context; }

	// Get a command queue by index, create it if doesn't exist
	cl_command_queue getCmdQueue(int index);

	// Print information of the platform
	int displayPlatformInfo();

	int displayDeviceInfo();

	int displayContextInfo();

	int displayAllInfo();

};


// Singleton instance
std::unique_ptr<clRuntime> clRuntime::instance;

clRuntime *clRuntime::getInstance()
{
	// Instance already exists
	if (instance.get())
		return instance.get();
	
	// Create instance
	instance.reset(new clRuntime());
	return instance.get();
}

clRuntime::clRuntime()
{
	cl_int err = 0;
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	checkOpenCLErrors(err, "Failed at clGetPlatformIDs");

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	checkOpenCLErrors(err, "Failed at clGetDeviceIDs");

	// Create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	checkOpenCLErrors(err, "Failed at clCreateContext");

}

clRuntime::~clRuntime()
{
	cl_int err = 0;

	if (context)
	{
		err = clReleaseContext(context);
		checkOpenCLErrors(err, "Failed at clReleaseContext");
	}

	for (auto &cmdQueue : cmdQueueRepo)
	{
		err = clReleaseCommandQueue(cmdQueue);
		checkOpenCLErrors(err, "Failed at clReleaseCommandQueue");
	}

}


int clRuntime::displayPlatformInfo(cl_platform_id plt_id, cl_platform_info plt_info)
{
	cl_int err;
	char platformInfo[1024];
	err = clGetPlatformInfo(plt_id, plt_info, sizeof(platformInfo),
		platformInfo, NULL);
	checkOpenCLErrors(err, "clGetPlatformInfo failed");
	std::cout << "\t" << platformInfo << std::endl;
}

int clRuntime::displayContextInfo(cl_context ctx, 
	cl_context_info ctx_info)
{
	// TODO
}

int clRuntime::displayPlatformInfo()
{
	
	std::cout << "Platform info:" << std::endl;
	displayPlatformInfo(platform, CL_PLATFORM_VENDOR);
	displayPlatformInfo(platform, CL_PLATFORM_VERSION);
	displayPlatformInfo(platform, CL_PLATFORM_PROFILE);
	displayPlatformInfo(platform, CL_PLATFORM_NAME);
	displayPlatformInfo(platform, CL_PLATFORM_EXTENSIONS);	
}

int clRuntime::displayDeviceInfo()
{
	cl_int err;
	
	// Get number of devices available
	cl_uint deviceCount = 0;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	checkOpenCLErrors(err, "Failed at clGetDeviceIDs");
	
	// Get device ids
	cl_device_id* deviceIds = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds, NULL);
	checkOpenCLErrors(err, "Failed at clGetDeviceIDs");
	
	// Print device index and device names
	std::cout << "Devices info:" << std::endl;
	for(cl_uint i = 0; i < deviceCount; ++i)
	{
		char deviceName[1024];
		err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, sizeof(deviceName),
			deviceName, NULL);
		checkOpenCLErrors(err, "Failed at clGetDeviceInfo");
		if (deviceIds[i] == device)
		{
			std::cout << "(*)\tDevice " << i << " = " << deviceName
				<<", Device ID = "<<deviceIds[i] << std::endl;

		}
		else 
		{
			std::cout << "\tDevice " << i << " = " << deviceName
				<<", Device ID = "<<deviceIds[i]<< std::endl;
		}
	}

	free(deviceIds);

	return 0;
}

int clRuntime::displayAllInfo()
{
	displayPlatformInfo();
	displayDeviceInfo();
}

cl_command_queue clRuntime::getCmdQueue(int index)
{
	cl_int err;

	if (index < cmdQueueRepo.size())
		return cmdQueueRepo[index];
	else
	{
		cl_command_queue cmdQ = clCreateCommandQueueWithProperties(context, device, 0, &err);
		checkOpenCLErrors(err, "Failed at clCreateCommandQueueWithProperties");
		return cmdQ;
	}
}

} // namespace clHelper

#endif
