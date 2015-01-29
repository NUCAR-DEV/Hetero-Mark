#ifndef CL_UTIL_H
#define CL_UTIL_H 

#include "clError.h"
#include "clFile.h"
#include "clRuntime.h"

namespace clHelper
{

cl_int clMemSet(cl_command_queue cmdQ, void *ptr, int value, size_t count)
{
        cl_int err;

        // Map
        err = clEnqueueSVMMap(cmdQ,
                              CL_TRUE,       // blocking map
                              CL_MAP_WRITE,
                              ptr,
                              count,
                              0, 0, 0
                              );
        checkOpenCLErrors(err, "Failed to clEnqueueSVMMap");

        // Set
        memset(ptr, value, count);

        // Unmap
        err = clEnqueueSVMUnmap(cmdQ, ptr, 0, 0, 0);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMUnmap");

        return err;
}
	
}

#endif
