#include "clUtil.h"

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

clUtil::~clUtil()
{

}

int clUtil::clInit()
{
	return 0;
}

