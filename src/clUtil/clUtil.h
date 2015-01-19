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

#endif
