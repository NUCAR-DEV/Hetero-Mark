#ifndef HMM_H
#define HMM_H

#include "clUtil.h"

using namespace clHelper;

class HMM
{
	clRuntime *runtime;
	clFile *file;

	int N;

	void Release();
public:
	HMM(int N);
	~HMM();

	void Param();
	void Forward();
	void Backward();
	void BaumWelch();
};

#endif
