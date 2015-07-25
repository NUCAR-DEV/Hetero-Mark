#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>

#define BILLION 1000000000L

int main(int argc, const char * argv[])
{
  uint64_t diff;
  struct timespec start, end;

  FILE *cl_code = fopen("kernel.cl", "r");
  if (cl_code == NULL) { printf("\nerror: clfile\n"); return(1); }
  char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
  int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
  fclose(cl_code);
  size_t source_length = strlen(source_str);
  
  cl_int err;
  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  cl_program program;
  
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) { printf("platformid %i", err); return 1; }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) { printf("deviceid %i", err); return 1; }

  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) { printf("createcontext %i", err); return 1; }

  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  if (err != CL_SUCCESS) { printf("commandqueue %i", err); return 1; }

  program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_length, &err);
  if (err != CL_SUCCESS) { printf("createprogram %i", err); return 1; }

  err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL2.0", NULL, NULL);
  if (err != CL_SUCCESS) { printf("buildprogram ocl20 %i", err); }

  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);
    return 1;
  }

  int i;
  int sz = 10;

  int *indata = (int *)malloc(sizeof(int)*sz);
  int *svm = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(int)*sz, 0);
    
  //  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */
 
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm, sizeof(int)*sz, 0, 0, 0);

  if (err != CL_SUCCESS) { printf("enqueuesvmmap ocl20 %i", err); }

  for (i = 0; i < sz; i++) { svm[i] = 2; }
  
  //  memcpy(&svm, &indata, sizeof(int)*sz);

  err = clEnqueueSVMUnmap(queue, svm, 0, 0, 0);

  if (err != CL_SUCCESS) { printf("enqueueunmap ocl20 %i", err); }
  
  //  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm, sizeof(int)*sz, 0, 0, 0);

  printf("\n Data \n");

  for (i=0; i<sz; i++) {
    printf("%d", svm[i]);
  }
  
  err = clEnqueueSVMUnmap(queue, svm, 0, 0, 0);
  err = clFinish(queue);
  clSVMFree(context, svm);
  free(indata);

  printf("--Done--");
  //  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  //  printf("\n Test 1 done, time: %llu nanoseconds\n", (long long unsigned int) diff);

  //  clReleaseCommandQueue(queue);
  //  clReleaseContext(context);
}
  /*---- */

  // int *svmnew = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(int)*10000, 0);
  // int *dummysvmnew = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(int)*10000, 0);
 
  // clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  // err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svmnew, sizeof(int)*10000, 0, 0, 0);
  // err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, dummysvmnew, sizeof(int)*10000, 0, 0, 0);

  // if (err != CL_SUCCESS) { printf("enqueuesvmmap ocl20 %i", err); }
  // for (i = 0; i < 10000; i++) { 
  //   memcpy(&svmnew[i], &indata[i], sizeof(int)); 
  //   memcpy(&dummysvmnew[i], &indata[i], sizeof(int)); 
  // }

  // err = clEnqueueSVMUnmap(queue, svmnew, 0, 0, 0);
  // err = clEnqueueSVMUnmap(queue, dummysvmnew, 0, 0, 0);

  // if (err != CL_SUCCESS) { printf("enqueueunmap ocl20 %i", err); }
  
  // clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  // clSVMFree(context, svmnew);
  // clSVMFree(context, dummysvmnew);
  
  // diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  // printf("\n Test 2 done, time: %llu nanoseconds\n", (long long unsigned int) diff);

