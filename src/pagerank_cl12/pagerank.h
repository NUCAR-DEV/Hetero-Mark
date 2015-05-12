#ifndef PageRank_H
#define PageRank_H

#include <clUtil.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <time.h>

using namespace clHelper;
using namespace std;

class PageRank
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
	void FillBuffer();
	void FillBufferCpu();
	void ExecKernel();

	void FreeKernel();
	void FreeBuffer();
	void ReadBuffer();

	void ReadCsrMatrix();
	void ReadDenseVector();
	void Print();
	void PageRankCpu();
	float* GetEigenV();




	string fileName1;
	string fileName2;
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
	ifstream csrMatrix;
	ifstream denseVector;
	size_t global_work_size[1];
	size_t local_work_size[1];
	int workGroupSize;

public:
	PageRank();
	PageRank(string, string);
	~PageRank();

	void Run();
	void CpuRun();
	void Test();
	
};

#endif
