#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>
#include <memory>

#include "iir.h"
#include "kernels.h"

#define BILLION 1000000000L

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
}

void ParIIR::Init()
{
	InitParam();
	InitBuffers();
}

void ParIIR::InitParam()
{
	// empty	
	channels = 64;
	c = 3.f;
}

void ParIIR::InitBuffers()
{
	int i, j;

	// Create the input and output arrays in device memory for our calculation
	X = (float *)malloc(sizeof(float) * len);
	gpu_Y = (float *)malloc(sizeof(float) * len * channels);
	nsec = (float *)malloc(sizeof(float) * ROWS);
	dsec = (float *)malloc(sizeof(float) * ROWS);

	size_t bytes = sizeof(float) * len;

	// input
	for (i=0; i<len; i++){
		X[i] = 0.1f;
	}
	
	for(i=0; i<ROWS; i++){
		nsec[i] = 0.00002f;
		dsec[i] = 0.00005f;
	}

	// cpu output: single channel
	cpu_y= (float*) malloc(bytes);

}

void ParIIR::CleanUpBuffers()
{
}

void ParIIR::multichannel_pariir()
{
	SNK_INIT_LPARM(lparm, 0);
	lparm->ndim = 1;
	lparm->ldims[0] = ROWS;
	lparm->gdims[0] = channels * ROWS;

	IIR(len, c, nsec, dsec, sizeof(float) * 512, X, gpu_Y, lparm);
}

void ParIIR::compare()
{
	//----------------------------------------------
	//	Compute CPU results
	//----------------------------------------------
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
		out = c * X[i];

		for(j=0; j<ROWS; j++)
		{
			unew = X[i] - (ds[j*2] * u[j*2] + ds[j*2+1] * u[j*2+1]);
			u[j*2+1] = u[j * 2];
			u[j*2] = unew;
			out = out + (u[j*2] * ns[j*2] + u[j*2 + 1] * ns[j*2 + 1]);
		}

		cpu_y[i] = out;
		//printf("cpu: %f\n", out);
	}

	//--------------------------------------------
	//	Compare CPU and GPU results
	//--------------------------------------------
	int success;

	int chn;
	for(chn=0; chn<channels; chn++)
	{
		size_t start = chn * len;

		for(i=0; i<len; i++)
		{
			if(abs(cpu_y[i] - gpu_Y[i + start]) > 0.001)	
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
	uint64_t diff;
	struct timespec start, end;
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

	//	double start = time_stamp();
	clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

	parIIR->Run();

	clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("Total elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

	//        double end = time_stamp();

	//        printf("Total time = %f ms\n", end - start);

	return 0;
}



