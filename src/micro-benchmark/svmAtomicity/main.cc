#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>
#include <CL/cl.h>

#define BILLION 1000000000L

#define CHECK_STATUS( status, message )		\
  if(status != CL_SUCCESS)			\
    {						\
      printf( message);				\
      printf( "\n" );				\
      return 1;					\
    }

/** Define custom constants*/
#define MAX_SOURCE_SIZE (0x100000)

cl_int err;
int *input = NULL;

int main(int argc , char** argv) {

  uint64_t diff;
  struct timespec start, end;
  
  // Define custom variables
  int i,count;
  int local;
  int sz = 1;

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("kernel.cl", "r");
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

  printf("\n No of Devices %d",ret_num_platforms );

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
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
  // Allocate SVM buffers
  input = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, sz*sizeof(int), 0);

  // Map SVM buffers for writing
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, input, sz*sizeof(int), 0, 0, 0);
  if (err != CL_SUCCESS) { printf("Error clEnqueueSVMMap input :: %i", err); exit(1); }   
    
  // Initialize the input data 
  input[0] = 2;
    
  // Done writing, unnmap SVM buffers
  err = clEnqueueSVMUnmap(queue, input, 0, 0, 0);
  if (err != CL_SUCCESS) { printf("Error clEnqueueSVMUnmap input :: %i", err); exit(1); }
    
  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,
						 (const char **)&source_str, (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id, "-I ./ -cl-std=CL2.0", NULL, NULL);

  CHECK_STATUS( ret,"Error: Build Program\n");

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "CLRunner", &ret);
  CHECK_STATUS( ret,"Error: Create kernel. (clCreateKernel)\n");

  // Set the arguments of the kernel
  // Use clSetKernelArgSVMPointer to set SVM pointers as arguments
  ret = clSetKernelArgSVMPointer(kernel, 0, input);
  
  // Decide the local group size formation
  size_t globalThreads[1]={1};
  size_t localThreads[1]={1};
  cl_command_type cmdType;
  count = 0;

 
  // Custom item size based on current algorithm
  size_t global_item_size = 1; 
  size_t local_item_size = 1;
  // Execute the OpenCL kernel on the list
  cl_event event;
  
  /* measure monotonic time */
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

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
  
  // Flush memory buffers
  ret = clFlush(queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  clSVMFree(context, input);
  ret = clReleaseCommandQueue(queue);
  ret = clReleaseContext(context);
  
  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

  return 0;
}
