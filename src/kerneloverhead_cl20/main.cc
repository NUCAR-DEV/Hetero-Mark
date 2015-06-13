#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)
#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>

#include <CL/cl.h>

#define BILLION 1000000000L

int main(int argc, const char * argv[])
{
  /*  srand(time(NULL));
  clock_t c_main_start, c_main_stop, c_test_start, c_test_stop;
  c_main_start = clock(); */
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

  
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) { printf("buildprogram ocl12 %i", err); }
  
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);
    return 1;
  }
  int i;
  cl_kernel kernel = clCreateKernel(program, "CLRunner", &err);
  if (err != CL_SUCCESS) { printf("createkernel %i", err); }
  cl_event event;
  const size_t local = 1;
  const size_t global = 1;

  /*  c_test_start = clock(); */
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  for (int i = 0; i < 100000; i++)
    {
      err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
      if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
      clFinish(queue);
    }
  /* c_test_stop = clock();
  diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
  printf("\n\tTest-1 done, time: %f ms", diff); */
  
  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("\n\tTest-1 done, time: %llu nanoseconds\n", (long long unsigned int) diff);
  
  // async
  //  c_test_start = clock();
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  for (int i = 0; i < 100000; i++)
    {
      err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
      if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
    }
  clFinish(queue);
  /*   c_test_stop = clock();
  diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
  printf("\n\tTest-2 done, time: %f ms", diff); */
  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("\n\tTest-2 done, time: %llu nanoseconds\n", (long long unsigned int) diff);

  clReleaseContext(context);
  clReleaseCommandQueue(queue);
  printf("\n");
}

