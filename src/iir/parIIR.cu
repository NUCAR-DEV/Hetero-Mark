// System includes
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h> 

// Provide Debugging Functionality
#include "cuPrintf.cu"

#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
		blockIdx.y*gridDim.x+blockIdx.x,\
		threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
		__VA_ARGS__)
#endif

#define ROWS 256  // num of parallel subfilters
#define DEB 0	  // compare cpu and gpu results
#define TIMING 1  // measure the kernel execution time


__constant__ float2 NSEC[ROWS];
__constant__ float2 DSEC[ROWS];

// Parallel IIR: CPU 
void cpu_pariir(float *x, float *y, float *ns, float *dsec, float c, int len);

// Check the results from CPU and GPU 
void check(float *cpu, float *gpu, int len, int tot_chn);


template <int blockSize>
__global__ void GpuParIIR (float *x, int len, float c, float *y)
{
	extern __shared__ float sm[];
	float *sp = &sm[ROWS];

	int tid = threadIdx.x;
	//int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	// & 0x20
	int lane_id = tid % 32; // warp size 32 for +3.5 device
	int warp_id = tid / 32;

	int ii, jj, kk;

	float2 u = make_float2(0.0f);
	float unew;
	float y0;

	// block size : ROWS
	// each thread fetch input x to shared memory
	for(ii=0; ii<len; ii+=ROWS)
	{
		sm[tid] = x[tid + ii];	

		__syncthreads();

		// go through each x in shared memory 
		for(jj=0; jj<ROWS; jj++)	
		{
			unew = sm[jj] - dot(u, DSEC[tid]);				
			u = make_float2(unew, u.x);
			y0 = dot(u, NSEC[tid]);
		
			// sum v across current block
			#pragma unroll
			for(kk=1; kk<32; kk<<=1)  
			{
				y0 += __shfl_xor(y0, kk, 32); 
			}

			if(lane_id == 0)
			{
				sp[warp_id] = y0;
			}

			__syncthreads();

			if(blockSize == 256 && warp_id == 0)
			{
				if(lane_id < 8)
				{
					float warp_sum = sp[lane_id];

					warp_sum += __shfl_xor(warp_sum, 1, 32);  // ? 32
					warp_sum += __shfl_xor(warp_sum, 2, 32);  // ? 32
					warp_sum += __shfl_xor(warp_sum, 4, 32);  // ? 32

					if(lane_id == 0){
						// channel starting postion: blockId.x * len
						uint gid = __mul24(blockIdx.x , len) + ii + jj;
						y[gid] = warp_sum + sm[jj] * c;	
					}
				}

			}
		}
	
	}

}


int main(int argc, char *argv[])
{
	if(argc != 2){
		printf("Missing the length of input!\nUsage: ./parIIR Len\n");
		exit(EXIT_FAILURE);	
	}

	int i, j;
	int channels = 64;

	int len = atoi(argv[1]); // signal length 

	size_t bytes = sizeof(float) * len;

	// input
	float *x= (float*) malloc(bytes);
	for (i=0; i<len; i++){
		x[i] = 0.1f;
	}

	// output: multi-channel from GPU
	float *gpu_y= (float*) malloc(bytes * channels);

	// cpu output:
	float *cpu_y= (float*) malloc(bytes);

	float c = 3.0;

	// coefficients
	float *nsec, *dsec;
	nsec = (float*) malloc(sizeof(float) * 2 * ROWS); // numerator
	dsec = (float*) malloc(sizeof(float) * 3 * ROWS); // denominator

	for(i=0; i<ROWS; i++){
		for(j=0; j<3; j++){
			dsec[i*3 + j] = 0.00002f;
		}
	}

	for(i=0; i<ROWS; i++){
		for(j=0; j<2; j++){
			nsec[i*2 + j] = 0.00005f;
		}
	}

	// compute the cpu results
	cpu_pariir(x, cpu_y, nsec, dsec, c, len);

	int warpsize = 32;
	int warpnum = ROWS/warpsize;

	// vectorize the coefficients
	float2 *vns, *vds;
	vns = (float2*) malloc(sizeof(float2) * ROWS);
	vds = (float2*) malloc(sizeof(float2) * ROWS); 

	for(i=0; i<ROWS; i++){
		vds[i] = make_float2(0.00002f);
		vns[i] = make_float2(0.00005f);
	}

	// timer
	cudaEvent_t start, stop;

	// device memory
	float *d_x;
	cudaMalloc((void **)&d_x, bytes);

	float *d_y;
	cudaMalloc((void **)&d_y, bytes * channels);

	// copy data to constant memory
	cudaMemcpyToSymbol(NSEC, vns, sizeof(float2)*ROWS, 0,
			           cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DSEC, vds, sizeof(float2)*ROWS, 0, 
			           cudaMemcpyHostToDevice);

	cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

#if TIMING
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start timer
	cudaEventRecord(start, 0);
#endif

	// kernel
	GpuParIIR <ROWS>
    <<< channels, ROWS, sizeof(float) * (ROWS + warpnum) >>> (d_x, len, c, d_y);

#if TIMING
	// end timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float et;
	cudaEventElapsedTime(&et, start, stop);
	printf ("ElapsetTime = %f (s)\n", et/1000.f);
#endif


	cudaMemcpy(gpu_y, d_y, bytes * channels, cudaMemcpyDeviceToHost);

	check(cpu_y, gpu_y, len, channels);

	// release
	cudaFree(d_x);
	cudaFree(d_y);

	free(x);
	free(gpu_y);
	free(cpu_y);
	free(dsec);
	free(nsec);
	free(vds);
	free(vns);

}


void cpu_pariir(float *x, float *y, float *ns, float *dsec, float c, int len)
{
	int i, j;
	float out;
	float unew;

	float *ds = (float*) malloc(sizeof(float) * ROWS * 2);	

	// internal state
	float *u = (float*) malloc(sizeof(float) * ROWS * 2);
	memset(u, 0 , sizeof(float) * ROWS * 2);

	for(i=0; i<ROWS; i++)
	{
		ds[i * 2]     = dsec[3 * i + 1];
		ds[i * 2 + 1] = dsec[3 * i + 2];
	}

	for(i=0; i<len; i++)
	{
		out = c * x[i];

		for(j=0; j<ROWS; j++)
		{
			unew = x[i] - (ds[j*2] * u[j*2] + ds[j*2+1] * u[j*2+1]);
			u[j*2+1] = u[j * 2];
			u[j*2] = unew;
			out = out + (u[j*2] * ns[j*2] + u[j*2 + 1] * ns[j*2 + 1]);
		}

		y[i] = out;
	}

	free(ds);
	free(u);
}


void check(float *cpu, float *gpu, int len, int tot_chn)
{
	int i;
	int chn;
	uint start;
	int success = 1;

	
	for(chn=0; chn<tot_chn; chn++)
	{
		start = chn * len;

		for(i=0; i<len; i++)
		{
			if(cpu[i] - gpu[i + start] > 0.0001)	
			{
				puts("Failed!");
				success = 0;
				break;
			}
		}
	}

	if(success)
		puts("Passed!");

#if DEB
	for(i=0; i<len; i++)
	{
		printf("[%d]\t cpu=%f \t gpu=%f\n", i, cpu[i], gpu[i]);	
	}
#endif
}
