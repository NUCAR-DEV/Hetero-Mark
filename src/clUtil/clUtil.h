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
	cl_context       context;
	cl_device_id*    devices;
	cl_command_queue commandQueue;

public:
	// Destructor
	~clUtil();

	static clUtil *getInstance();

	int clInit();

};

#endif
