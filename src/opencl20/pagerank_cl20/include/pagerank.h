#ifndef PageRank_H
#define PageRank_H

#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <time.h>/* for clock_gettime */

#include "src/common/cl_util/cl_util.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <cmath>
#include "src/common/benchmark/benchmark.h"

#define BILLION 1000000000L

using namespace clHelper;
//using namespace std;

class PageRank : public Benchmark
{
private:
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue;

	cl_program       program;
	cl_kernel        kernel;
        cl_int 		 err;

	void InitKernel();
	void InitBuffer();
	void InitBufferCpu();
	void InitBufferGpu();
	void FillBuffer();
	void FillBufferCpu();
	void FillBufferGpu();
	void ExecKernel();

	void FreeKernel();
	void FreeBuffer();
	void ReadBuffer();

	void ReadCsrMatrix();
	void ReadDenseVector();
	void PageRankCpu();

	std::string fileName1;
	std::string fileName2;
	int nnz;
	int nr;
	int* rowOffset;
	int* col;
	float* val;
	float* vector;
	float* eigenV;
	cl_mem d_rowOffset;
	cl_mem d_col;
	cl_mem d_val;
	cl_mem d_vector;
	cl_mem d_eigenV;
	std::ifstream csrMatrix;
	std::ifstream denseVector;
	size_t global_work_size[1];
	size_t local_work_size[1];
	int workGroupSize;
	int maxIter;
	int isVectorGiven;

public:
	PageRank();

	~PageRank();
	void SetInitialParameters(std::string fName1, std::string fName2);
	void SetInitialParameters(std::string fName1);
	void Initialize() override {}
	void Run() override;
	void Verify() override {}
	void Cleanup() override {}
	void Summarize() override {}

	void CpuRun();
	void Test();
	float* GetEigenV();
	void Print();
	void PrintOutput();
	int GetLength();
	float abs(float);
	
};

#endif
