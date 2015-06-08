#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <CL/cl.h>
//#ifdef GPUPROF
//#include "inc/GPUPerfAPI.h"
//#include <dlfcn.h>
//#endif


#define CHECK_STATUS( status, message )   \
		if(status != CL_SUCCESS) \
		{ \
			printf( message); \
			printf( "\n" ); \
			return 1; \
		}

/** Define custom constants*/
#define MAX_SOURCE_SIZE (0x100000)

cl_int err;
cl_uint numTap = 0;
cl_uint numData = 0;		// Block size
cl_uint numTotalData = 0;
cl_uint numBlocks = 0;		// Number of blocks
cl_float* input = NULL;
cl_float* output = NULL;
cl_float* coeff = NULL;
cl_float* temp_output = NULL;


int main(int argc , char** argv) {
	// Timer for benchmark use
	clock_t c_start, c_stop;	
	c_start = clock();

	// Define custom variables
	int i,count;
	int local;

	if (argc < 3)	// Inavlid arguments
	{
		printf(" Usage : ./auto_exec.sh <numBlocks> <numData>\n");
		exit(0);
	}
	if (argc > 1)	// Read arguments into value
	{
		numBlocks = atoi(argv[1]);
		numData = atoi(argv[2]);
	}

	/** Declare the Filter Properties */
	numTap = 1024;
	numTotalData = numData * numBlocks;
	local = 64;

	printf("FIR Filter <OpenCL 2.0 Modification CM>\n Data Samples : %d \n NumBlocks : %d \n Local Workgroups : %d\n", numData,numBlocks,local);

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("fir_Kernels.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1,
			&device_id, &ret_num_devices);

	printf("/n No of Devices %d",ret_num_platforms );

	// Get platform information
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
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	// Allocate SVM buffers
    input = (cl_float *)clSVMAlloc(context, CL_MEM_READ_ONLY, numTotalData*sizeof(cl_float), 0);
    output = (cl_float *)clSVMAlloc(context, CL_MEM_READ_WRITE, numTotalData*sizeof(cl_float), 0);
    coeff = (cl_float *)clSVMAlloc(context, CL_MEM_READ_ONLY, numTap*sizeof(cl_float), 0);
    temp_output = (cl_float *)clSVMAlloc(context, CL_MEM_READ_WRITE, (numData+numTap-1)*sizeof(cl_float), 0);
    
    // Map SVM buffers for writing
    err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, input, numTotalData*sizeof(cl_float), 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMMap input :: %i", err); exit(1); }   
    err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, output, numTotalData*sizeof(cl_float), 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMMap output :: %i", err); exit(1); }  
    err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, coeff, numTap*sizeof(cl_float), 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMMap coeff :: %i", err); exit(1); }
    err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, temp_output, (numData+numTap-1)*sizeof(cl_float), 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMMap temp_output :: %i", err); exit(1); }
    
	// Initialize the input data 
    for( i=0;i<numTotalData;i++ )
    {
        input[i] = 8;
        output[i] = 99;
    }
    
    for( i=0;i<numTap;i++ )
        coeff[i] = 1.0/numTap;
    
    for( i=0; i<(numData+numTap-1); i++ )
        temp_output[i] = 0.0;
    
#if 1
    // Read the input file
    FILE *fip;
    i=0;
    fip = fopen("temp.dat","r");
    while(i<numTotalData)
    {
        int res = fscanf(fip,"%f",&input[i]);
        i++;
    }
    fclose(fip);
    
#if 0
    printf("\n The Input:\n");
    i = 0;
    while( i<numTotalData )
    {
        printf( "%f, ", input[i] );
        
        i++;
    }
#endif
#endif
    
    // Done writing, unnmap SVM buffers
    err = clEnqueueSVMUnmap(queue, input, 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMUnmap input :: %i", err); exit(1); }
    
    err = clEnqueueSVMUnmap(queue, output, 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMUnmap output :: %i", err); exit(1); }
    
    err = clEnqueueSVMUnmap(queue, coeff, 0, 0, 0);
    if (err != CL_SUCCESS) { printf("Error clEnqueueSVMUnmap coeff :: %i", err); exit(1); }

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, "-I ./ -cl-std=CL2.0", NULL, NULL);

	CHECK_STATUS( ret,"Error: Build Program\n");

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "FIR", &ret);
	CHECK_STATUS( ret,"Error: Create kernel. (clCreateKernel)\n");

	// Set the arguments of the kernel
	// Use clSetKernelArgSVMPointer to set SVM pointers as arguments
	ret = clSetKernelArgSVMPointer(kernel, 0, output); 
	ret = clSetKernelArgSVMPointer(kernel, 1, coeff);	
	ret = clSetKernelArgSVMPointer(kernel, 2, temp_output);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&numTap); // Not a SVM pointer

	// Decide the local group size formation
	size_t globalThreads[1]={numData};
	size_t localThreads[1]={128};
	cl_command_type cmdType;
	count = 0;

	// FIR Loop
	double execTimeMs = 0.0f; 
	while( count < numBlocks )
	{	
		// Custom item size based on current algorithm
		size_t global_item_size = numData; 
		size_t local_item_size = numData;
		// Execute the OpenCL kernel on the list
        cl_event event;
			ret = clEnqueueNDRangeKernel(
					queue,
					kernel,
					1,
					NULL,
					globalThreads,
					localThreads,
					0,
					NULL,
					&event);


			CHECK_STATUS( ret,"Error: Range kernel. (clCreateKernel)\n");
			ret = clWaitForEvents(1, &event);

        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        double time = (cl_double)((end - start)/1e-06); 
        execTimeMs += time;
        printf("%lu %lu = %lu\n", end, start, end - start);
		count ++;
	}
	printf("\nKernel exec time: %f\n", execTimeMs);

	// Flush memory buffers
	ret = clFlush(queue);
	ret = clFinish(queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	clSVMFree(context, input);
	clSVMFree(context, output);
	clSVMFree(context, coeff);
	clSVMFree(context, temp_output);
	ret = clReleaseCommandQueue(queue);
	ret = clReleaseContext(context);

	// Timer for benchmark use
	c_stop = clock();
	float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC) * 1000;
	printf("\nDone! - Time taken: %f\n", diff);
	return 0;
}
