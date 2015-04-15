#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <clUtil.h>
#include <CL/cl_platform.h> // cl_float2

#include "parIIR.h"

void cpu_pariir(float *x, float *y, float *ns, float *dsec, float c, int len);

ParIIR::ParIIR(int len)
{
	this->len = len;	
}

ParIIR::~ParIIR()
{
	CleanUp();
}

void ParIIR::CleanUp()
{
	CleanUpBuffers();
	CleanUpKernels();
}

void ParIIR::Init()
{
	InitParam();
	InitCL();	
	InitKernels();
	InitBuffers();
}

void ParIIR::InitParam()
{
	// empty	
	channels = 64;
	c = 3.f;
}

void ParIIR::InitCL()
{
	// Init OCL context
	runtime    = clRuntime::getInstance();

	// OpenCL objects get from clRuntime class release themselves automatically, 
	// no need to clRelease them explicitly
	platform   = runtime->getPlatformID();
	device     = runtime->getDevice();
	context    = runtime->getContext();

	cmdQueue   = runtime->getCmdQueue(0);
	//cmdQueue_1 = runtime->getCmdQueue(1);

	// Helper to read kernel file
	file       = clFile::getInstance();
}

void ParIIR::InitKernels()
{
	cl_int err;	
	
	file->open("kernel.cl");

	// Create program
	const char *source = file->getSourceChar();

	program = clCreateProgramWithSource(context, 
                                        1, 
			                            (const char **)&source, 
                                        NULL, 
                                        &err);
	checkOpenCLErrors(err, "Failed to create Program with source...\n");

	// Build program 
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	checkOpenCLErrors(err, "Failed to build program...\n");


	kernel_pariir = clCreateKernel(program, "ParIIR", &err);
	checkOpenCLErrors(err, "Failed to create kernel ParIIR\n");
}

void ParIIR::InitBuffers()
{
	cl_int err;

	int i, j;

	// Create the input and output arrays in device memory for our calculation
	d_X  	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*len, NULL, NULL);
	d_Y  	= clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*len*channels, NULL, NULL);
	d_nsec 	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_float2)*ROWS, NULL, NULL);
	d_dsec 	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_float2)*ROWS, NULL, NULL);

	size_t bytes = sizeof(float) * len;

	// input
	h_X = (float*) malloc(bytes);
	for (i=0; i<len; i++){
		h_X[i] = 0.1f;
	}
	
	// coefficients
	nsec = (cl_float2*) malloc(sizeof(cl_float2) * ROWS); // numerator
	dsec = (cl_float2*) malloc(sizeof(cl_float2) * ROWS); // denominator

	for(i=0; i<ROWS; i++){
		//nsec[i].x = nsec[i].y = 0.00002f;
		//dsec[i].x = dsec[i].y = 0.00005f;
		nsec[i] = {0.00002f, 0.00002f};
		dsec[i] = {0.00005f, 0.00005f};
	}

	// cpu output: single channel
	cpu_y= (float*) malloc(bytes);

	// gpu output: multi channel
	h_Y = (float *) malloc(bytes*channels);

	//err = clEnqueueWriteBuffer(cmdQueue, d_nsec, CL_TRUE, 0, sizeof(float)*row*2, nsec, 0, NULL, NULL);
	//checkOpenCLErrors(err, "Faile to write d_nsec to device.\n");

	err = clEnqueueWriteBuffer(cmdQueue, d_X, CL_TRUE, 0, bytes, h_X, 0, NULL, NULL);
	checkOpenCLErrors(err, "Faile to write h_X to device.\n");

	err = clEnqueueWriteBuffer(cmdQueue, d_nsec, CL_TRUE, 0, sizeof(cl_float2)*ROWS, nsec, 0, NULL, NULL);
	checkOpenCLErrors(err, "Faile to write nsec to device.\n");

	err = clEnqueueWriteBuffer(cmdQueue, d_dsec, CL_TRUE, 0, sizeof(cl_float2)*ROWS, dsec, 0, NULL, NULL);
	checkOpenCLErrors(err, "Faile to write dsec to device.\n");
}

void ParIIR::CleanUpBuffers()
{
	clReleaseMemObject(d_X);
	clReleaseMemObject(d_Y);
}

void ParIIR::CleanUpKernels()
{
	clReleaseKernel(kernel_pariir);
}

void ParIIR::multichannel_pariir()
{
	cl_int err;

	size_t localSize = ROWS; // multiple of 64
	size_t globalSize = channels * localSize;

	// Set the arguments to our compute kernel
	err  = clSetKernelArg(kernel_pariir, 0, sizeof(int), &len);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 1, sizeof(float), &c);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 2, sizeof(cl_mem), &d_nsec);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 3, sizeof(cl_mem), &d_dsec);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 4, sizeof(float) * 512, NULL);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 5, sizeof(cl_mem), &d_X);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err  = clSetKernelArg(kernel_pariir, 6, sizeof(cl_mem), &d_Y);
	checkOpenCLErrors(err, "Failed at clSetKernelArg");

	err = clEnqueueNDRangeKernel(cmdQueue, 
                                 kernel_pariir, 
								 1, 
								 NULL, 
                                 &globalSize, 
								 &localSize, 
                                 0, 
								 NULL, 
								 NULL);

	checkOpenCLErrors(err, "Failed to execute kernel.\n");
}

void ParIIR::compare()
{
	// copy gpu results back	
	clEnqueueReadBuffer(cmdQueue, d_Y, CL_TRUE, 0, sizeof(float)*len*channels, h_Y, 0, NULL, NULL);


	//-------------------------------------------------------------------------------------------//
	//							Compute CPU results
	//-------------------------------------------------------------------------------------------//
	float *ds = (float*) malloc(sizeof(float) * ROWS * 2);	
	float *ns = (float*) malloc(sizeof(float) * ROWS * 2);	

	// internal state
	float *u = (float*) malloc(sizeof(float) * ROWS * 2);
	memset(u, 0 , sizeof(float) * ROWS * 2);

	float out, unew;

	int i, j;

	for(i=0; i<ROWS; i++)
	{
		ds[i*2] = ds[i*2 + 1] = 0.00005f;
		ns[i*2] = ns[i*2 + 1] = 0.00002f;
	}

	for(i=0; i<len; i++)
	{
		out = c * h_X[i];

		for(j=0; j<ROWS; j++)
		{
			unew = h_X[i] - (ds[j*2] * u[j*2] + ds[j*2+1] * u[j*2+1]);
			u[j*2+1] = u[j * 2];
			u[j*2] = unew;
			out = out + (u[j*2] * ns[j*2] + u[j*2 + 1] * ns[j*2 + 1]);
		}

		cpu_y[i] = out;
		//printf("cpu: %f\n", out);
	}

	//-------------------------------------------------------------------------------------------//
	//							Compare CPU and GPU results
	//-------------------------------------------------------------------------------------------//

	int success;

	int chn;
	for(chn=0; chn<channels; chn++)
	{
		size_t start = chn * len;

		for(i=0; i<len; i++)
		{
			if(abs(cpu_y[i] - h_Y[i + start]) > 0.001)	
			{
				puts("Failed!");
				success = 0;
				break;
			}
		}
	}

	if(success)
		puts("Passed!");
}

void ParIIR::Run()
{
	printf("=>Initialize parameters.\n");
	Init();

	printf("      >> Start IIR on GPU.\n");	

    multichannel_pariir();
 	
	printf("      >> End IIR on GPU.\n");	

	// check results
	compare();

	printf("<=End program.\n");
}

//-----------------------------------------------------------------------------------------------//
// 	                                       Main Function
//-----------------------------------------------------------------------------------------------//
int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		printf("Missing the length of input!\nUsage: ./parIIR Len\n");
		exit(EXIT_FAILURE);	
	}

	int len = atoi(argv[1]);

	// compute the cpu results
	//cpu_pariir(x, cpu_y, nsec, dsec, c, len);

	// opencl 1.2
	std::unique_ptr<ParIIR> parIIR(new ParIIR(len));

	double start = time_stamp();
	parIIR->Run();
        double end = time_stamp();

        printf("Total time = %f ms\n", end - start);

	return 0;
}



