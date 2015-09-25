#ifndef AES_H
#define AES_H

#include "src/common/cl_util/cl_util.h"
#include <CL/cl.h>
#include "src/common/benchmark/benchmark.h"

using namespace clHelper;

class FIR : public Benchmark
{
private:
    cl_uint numTap = 0;
    cl_uint numData = 0;  // Block size
    cl_uint numTotalData = 0;
    cl_uint numBlocks = 0;
    cl_float* input = NULL;
    cl_float* output = NULL;
    cl_float* coeff = NULL;
    cl_float* temp_output = NULL;
    
public:
        FIR() {};
        ~FIR() {};

    void SetInitialParameters(int data, int blocks) { numBlocks = blocks; numData = data; }
    	void Initialize() override {}
	void Run() override;
	void Verify() override {}
	void Cleanup() override {}
	void Summarize() override {}
};

#endif
