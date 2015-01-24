#ifndef CL_CMD_QUEUE_REPO_H
#define CL_CMD_QUEUE_REPO_H

#include <memory>
#include <vector>
#include <CL/cl.h>

#include "clRuntime.h"
#include "clError.h"

namespace clHelper
{

// CmdQueueRepo cl_command_queue objects
class clCmdQueueRepo
{

	std::vector<cl_command_queue> cmdQueueRepo;

	// Instance of the singleton
	static std::unique_ptr<clCmdQueueRepo> instance;

	// Private constructor for singleton
	clCmdQueueRepo();
	
public:

	// Get singleton
	static clCmdQueueRepo *getInstance();

	~clCmdQueueRepo();

	// Add a cl_command_queue, no properties specified
	cl_command_queue add();

	// Remove a cl_command_queue
	int remove(cl_command_queue cmdQueue);

	// Get index of a cl_command_queue
	unsigned getIdx(cl_command_queue cmdQueue);
	
};

// Singleton instance
std::unique_ptr<clCmdQueueRepo> clCmdQueueRepo::instance;

clCmdQueueRepo *clCmdQueueRepo::getInstance()
{
	// Instance already exists
	if (instance.get())
		return instance.get();
	
	// Create instance
	instance.reset(new clCmdQueueRepo());
	return instance.get();
}

clCmdQueueRepo::clCmdQueueRepo()
{

}

// Destructor: automatically release command queue
clCmdQueueRepo::~clCmdQueueRepo()
{
	cl_int err;

	for (auto &cmdQueue : cmdQueueRepo)
	{
		if (cmdQueue)
		{
			err = clReleaseCommandQueue(cmdQueue);
			checkOpenCLErrors(err, "Failed at clReleaseCommandQueue");			
		}
	}
}

cl_command_queue clCmdQueueRepo::add()
{
	cl_context context = clRuntime::getInstance()->getContext();
	cl_device_id device = clRuntime::getInstance()->getDevice();

	cl_int err;
	cl_command_queue queue;
	
	queue = clCreateCommandQueue(context, device, 0, &err);
	checkOpenCLErrors(err, "Failed at clCreateCommandQueueWithProperties");

	return queue;
}

} // namespace clHelper


#endif
