#include <clUtil.h>

#include "shallowWater.h"

int shallowWater::setupCL()
{
	int ret;

	clUtil *clutil = clUtil::getInstance();
	ret = clutil->clInit();

	return ret;
}

int main(int argc, char const *argv[])
{
	
	return 0;
}