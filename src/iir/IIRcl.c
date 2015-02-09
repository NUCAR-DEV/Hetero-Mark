#include <stdio.h>
#include <stdlib.h>
// Next 2 not sure if needed
#include <time.h>
#include <string.h>
// OpenCL 1.2 lib
#include <CL/cl.h>

#define ROWS 256  // num of parallel subfilters
#define DEB 0	  // compare cpu and gpu results

// Parallel IIR: CPU 
// void cpu_pariir(cl_float *x, cl_float *y, cl_float *ns, cl_float *dsec, cl_float c, cl_int len);

// Check the results from CPU and GPU 
// void check(cl_float *cpu, cl_float *gpu, int len, int tot_chn);

int main(int argc, char *argv[])
{
	if(argc != 2){
		printf("Missing the length of input!\nUsage: ./parIIR Length\n");
		exit(0); // taken from FIR	
	}

	int i, j; // counters
	int channels = 64;

	cl_int len = atoi(argv[1]); // take signal length from arg input

	size_t bytes = sizeof(cl_float) * len; // easier for latter malloc

	// input
	cl_float *x= (cl_float*) malloc(bytes);
	for (i=0; i<len; i++){
		x[i] = 0.1f;
	}

	// output: multi-channel from GPU
	cl_float *gpu_y= (cl_float*) malloc(bytes * channels);

	// cpu output:
	cl_float *cpu_y= (cl_float*) malloc(bytes);

	cl_float c = 3.0;

	// coefficients
	cl_float *nsec, *dsec;
	 nsec = (cl_float*) malloc(sizeof(cl_float) * 2 * ROWS); // numerator
	dsec = (cl_float*) malloc(sizeof(cl_float) * 3 * ROWS); // denominator

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
	// cpu_pariir(x, cpu_y, nsec, dsec, c, len);

	int warpsize = 32;
	int warpnum = ROWS/warpsize;

	// vectorize the coefficients (cl_float2 also included in OpenCl)
	cl_float2 *vns, *vds;
	vns = (cl_float2*) malloc(sizeof(cl_float2) * ROWS);
	vds = (cl_float2*) malloc(sizeof(cl_float2) * ROWS); 

	for(i=0; i<ROWS; i++){
		vds[i] = (float2)(0.00002f); 	// make_float2 is CUDA specific
		vns[i] = (float2)(0.00005f);		// changed to (float2)
	}

	// OpenCL init starts here
	// Load the kernel source code into the array source_str

	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("IIR.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1,
			&device_id, &ret_num_devices);

	printf("/n No of Devices %d",ret_num_platforms );

	char *platformVendor;
	size_t platInfoSize;
	clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL,
			&platInfoSize);

	platformVendor = (char*)malloc(platInfoSize);

	clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, platInfoSize,
			platformVendor, NULL);
	printf("\tVendor: %s\n", platformVendor);
	free(platformVendor);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

	// Create memory buffers on the device for each vector
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &ret);
	cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes * channels, NULL, &ret);
	cl_mem NSEC = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2)*ROWS, NULL, &ret); //constants
	cl_mem DSEC = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2)*ROWS, NULL, &ret); //constants

	ret = clEnqueueWriteBuffer(command_queue, 
			NSEC,
			CL_TRUE,
			0,
			sizeof(float2)*ROWS,
			vns,
			0,
			0,
			&event);
	eventList->add(event);

	ret = clEnqueueWriteBuffer(command_queue, 
			DSEC,
			CL_TRUE,
			0,
			sizeof(float2)*ROWS,
			vds,
			0,
			0,
			&event);
	eventList->add(event);

	ret = clEnqueueWriteBuffer(command_queue, 
			d_x,
			CL_TRUE,
			0,
			bytes,
			x,
			0,
			0,
			&event);
	eventList->add(event);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	CHECK_STATUS( ret,"Error: Build Program\n");

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "IIR", &ret);
	CHECK_STATUS( ret,"Error: Create kernel. (clCreateKernel)\n");

	ret = clSetKernelArg(kernel, 0, bytes, (void *)&d_x);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&len);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&c);
	ret = clSetKernelArg(kernel, 3, bytes * channels, (void *)&d_y);
	ret = clSetKernelArg(kernel, 4, sizeof(float2)*ROWS, (void *)&NSEC);
	ret = clSetKernelArg(kernel, 5, sizeof(float2)*ROWS, (void *)&DSEC);

	// Decide the local group formation
	size_t globalThreads[1]={ROWS*channels}; 
	size_t localThreads[1]={ROWS};

	cl_command_type cmdType;
    cl_event event;
	ret = clEnqueueNDRangeKernel(
					queue,
					kernel,
					1,
					NULL,
					globalThreads,	// total number of threads
					localThreads,	// local number of threads (size of workgroup)
					0,
					NULL,
					&event);


			CHECK_STATUS( ret,"Error: Range kernel. (clCreateKernel)\n");
			ret = clWaitForEvents(1, &event);

	// Read back and write out result here
			
	// Release craps

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(d_x);
	ret = clReleaseMemObject(d_y);
	ret = clReleaseMemObject(NSEC);
	ret = clReleaseMemObject(DSEC);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(d_x);
	free(d_y);
	free(NSEC);
	free(DSEC);
}
