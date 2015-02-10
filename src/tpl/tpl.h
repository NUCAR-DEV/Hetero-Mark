#ifndef TPL_H
#define TPL_H

#include <clUtil.h>

using namespace clHelper;

class TPL
{
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue;

	cl_program       program;
	cl_kernel        kernel;

	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();

public:
	TPL();
	~TPL();

	void Run();
	
};

#endif
