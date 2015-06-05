#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>

int main(int argc, const char * argv[])
{
  srand(time(NULL));
  clock_t c_main_start, c_main_stop, c_test_start, c_test_stop;
  c_main_start = clock();
  
  FILE *cl_code = fopen("kernel.cl", "r");
  if (cl_code == NULL) { printf("\nerror: clfile\n"); return(1); }
  char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
  fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
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

  queue = clCreateCommandQueue(context, device, NULL, &err);
  if (err != CL_SUCCESS) { printf("commandqueue %i", err); return 1; }

  program = clCreateProgramWithSource(context, 1, &source_str, &source_length, &err);
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

  float diff = 0;
  int i;

  printf("\nmemtime int copy test:");
  int maxnum = 100;
  for (int x = 1; x <= 10; x++)
    {
      printf("\nTest %i, %i objects", x, x*maxnum);
      int *indata = (int *)malloc(sizeof(int)*x*maxnum);
      int *outdata = (int *)malloc(sizeof(int)*x*maxnum);
      for (i = 0; i < x*maxnum; i++) { indata[i] = rand(); }
  
      c_test_start = clock();
      cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x*maxnum*sizeof(int), indata, &err);
      if (err != CL_SUCCESS) { printf("createbuffer ocl12 %i", err); }
      err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, x*maxnum*sizeof(int), outdata, 0, NULL, NULL);
      if (err != CL_SUCCESS) { printf("readbuffer ocl12 %i", err); }
      clFinish(queue);
      c_test_stop = clock();
      clReleaseMemObject(buffer);

      for (i = 0; i < x*maxnum; i++) { if (indata[i] != outdata[i]) { printf("\nNote: Memory corruption occured during transfer(s)"); break; }}
      diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
      printf("\nTest %i done, time: %f ms", x, diff);
    }
  clReleaseContext(context);
  clReleaseCommandQueue(queue);
  printf("\n");
}

