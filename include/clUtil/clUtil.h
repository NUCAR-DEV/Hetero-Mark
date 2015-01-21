#ifndef CL_UTIL_H
#define CL_UTIL_H 

#include <memory>
#include <CL/cl.h>

#include "clError.h"

namespace clHelper
{




enum SVMCapability
{
	SVM_COARSE_GRAIN_BUFFER,
	SVM_FINE_GRAIN_BUFFER,
	SVM_FINE_GRAIN_SYSTEM,
	SVM_ATOMICS
};

class clUtil
{

private:
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

	int displayPlatformInfo(cl_platform_id platformId, 
		cl_platform_info info);

public:
	// Destructor
	~clUtil();

	// Get singleton
	static clUtil *getInstance();

	/// Getters
	cl_platform_id const getPlatformID() { return platform; }

	cl_context const getContext() { return context; }

	cl_device_id const getDevice() { return device; }

	cl_command_queue const getCmdQueue() { return cmdQueue; }

	// Check SVM support level
	bool const checkSVMCapability(enum SVMCapability cap);

	// Print information of the platform
	int displayPlatformInfo();

	int displayDeviceInfo();

	int displayInfo();

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
	cl_int err = 0;
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed at clGetPlatformIDs\n");
		exit(-1);
	}

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed at clGetDeviceIDs\n");
		exit(-1);
	}

	// Create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed at clCreateContext\n");
		exit(-1);
	}

	// Create a command queue 
	cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed at clCreateCommandQueueWithProperties\n");
		exit(-1);
	}

	// Mark as initialized
	isInit = true;
}

clUtil::~clUtil()
{
	if (isInit)
	{
		clReleaseCommandQueue(cmdQueue);
		clReleaseContext(context);
	}

}

bool const clUtil::checkSVMCapability(enum SVMCapability svmCap)
{
#ifdef CL_VERSION_2_0
	cl_device_svm_capabilities caps;
	cl_int err = clGetDeviceInfo(
		device,
		CL_DEVICE_SVM_CAPABILITIES,
		sizeof(cl_device_svm_capabilities),
		&caps,
		0);

	if (err != CL_SUCCESS)
		return false;
	
	switch(svmCap)
	{
	
	case SVM_COARSE_GRAIN_BUFFER:
		if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
		{
			printf("SVM level: coarse gain buffer\n");
			return true;
		}
		break;
	case SVM_FINE_GRAIN_BUFFER:
		if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
		{
			printf("SVM level: fine grain buffer\n");
			return true;
		}
		break;
	case SVM_FINE_GRAIN_SYSTEM:
		if (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
		{
			printf("SVM level: fine gain system\n");			
			return true;
		}
		break;
	case SVM_ATOMICS:
		if (caps & CL_DEVICE_SVM_ATOMICS)
		{
			printf("SVM level: atomics\n");
			return true;
		}
		break;
	default:
		break;
	}
#endif
	return false;

}

int clUtil::displayPlatformInfo(cl_platform_id platformId, cl_platform_info info)
{
	cl_int err;
	char platformInfo[1024];
	err = clGetPlatformInfo(platformId, info, sizeof(platformInfo),
		platformInfo, NULL);
	CHECK_OPENCL_ERROR(err, "clGetPlatformInfo failed");
	std::cout << "\t" << platformInfo << std::endl;
}

int clUtil::displayPlatformInfo()
{
	std::cout << "Platform info:" << std::endl;
	displayPlatformInfo(platform, CL_PLATFORM_VENDOR);
	displayPlatformInfo(platform, CL_PLATFORM_VERSION);
	displayPlatformInfo(platform, CL_PLATFORM_PROFILE);
	displayPlatformInfo(platform, CL_PLATFORM_NAME);
	displayPlatformInfo(platform, CL_PLATFORM_EXTENSIONS);	
}

int clUtil::displayDeviceInfo()
{
	cl_int err;
	
	// Get number of devices available
	cl_uint deviceCount = 0;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	CHECK_OPENCL_ERROR(err, "clGetDeviceIDs failed");
	cl_device_id* deviceIds = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
	
	// Get device ids
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds, NULL);
	CHECK_OPENCL_ERROR(err, "clGetDeviceIDs failed");
	
	// Print device index and device names
	std::cout << "Devices info:" << std::endl;
	for(cl_uint i = 0; i < deviceCount; ++i)
	{
		char deviceName[1024];
		err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, sizeof(deviceName),
			deviceName, NULL);
		CHECK_OPENCL_ERROR(err, "clGetDeviceInfo failed");
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


} // namespace clHelper

#endif
