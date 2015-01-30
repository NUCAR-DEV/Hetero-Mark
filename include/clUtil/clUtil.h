#ifndef CL_UTIL_H
#define CL_UTIL_H 

#include "clError.h"
#include "clFile.h"
#include "clRuntime.h"

namespace clHelper
{

#ifndef __NOT_IMPLEMENTED__
#define __NOT_IMPLEMENTED__ printf("Error: not implemented. Func %s Line %d\n", __FUNCTION__, __LINE__);
#endif
} // namespace clHelper

#endif
