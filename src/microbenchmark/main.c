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
  printf("\nCasSimpleOCLMemTests - buffer / kernel overhead test\nCarter McCardwell, NUCAR built April 13 2015\nPlease wait...");
  srand(time(NULL));
  clock_t c_main_start, c_main_stop, c_test_start, c_test_stop;
  c_main_start = clock();

  int runMode = atoi(argv[1]);
  int oclMode = atoi(argv[2]);

  if (runMode == 0 || oclMode == 0 || oclMode > 2 || oclMode < 1 || runMode > 3 || runMode < 0)
  {
    printf("\nArgument 1 specifies the mode - 1=Array Transfer, 2=Kernel Overhead, 3=memtime\nArgument 2 specifies the OCL mode - 1=1.2, 2=2.0\n");
    return 1;
  }

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

  switch (oclMode)
  {
    case 1:
      err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
      if (err != CL_SUCCESS) { printf("buildprogram ocl12 %i", err); }
      break;
    case 2:
      err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL2.0", NULL, NULL);
      if (err != CL_SUCCESS) { printf("buildprogram ocl20 %i", err); }
      break;
    default:
      return 1; break;
  }
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

  switch (runMode)
  {
    case 1:
      printf("\n100 int copy test:");
      int *indata = (int *)malloc(sizeof(int)*100);
      int *outdata = (int *)malloc(sizeof(int)*100);
      for (i = 0; i < 100; i++) { indata[i] = rand(); }
      if (oclMode == 1)
      {
        c_test_start = clock();
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 100*sizeof(int), indata, &err);
        if (err != CL_SUCCESS) { printf("createbuffer ocl12 %i", err); }
        err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, 100*sizeof(int), outdata, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("readbuffer ocl12 %i", err); }
        c_test_stop = clock();
        clReleaseMemObject(buffer);
      }
      else if (oclMode == 2)
      {
        c_test_start = clock();
        int *svm = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(int)*100, 0);
        err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm, sizeof(int)*100, 0, 0, 0);
        if (err != CL_SUCCESS) { printf("enqueuesvmmap ocl20 %i", err); }
        for (i = 0; i < 100; i++) { memcpy(&svm[i], &indata[i], sizeof(int)); }
        err = clEnqueueSVMUnmap(queue, svm, 0, 0, 0);
        if (err != CL_SUCCESS) { printf("enqueueunmap ocl20 %i", err); }
        err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm, sizeof(int)*100, 0, 0, 0);
        if (err != CL_SUCCESS) { printf("enqueusvmmap2 ocl20 %i", err); }
        for (i = 0; i < 100; i++) { memcpy(&outdata[i], &svm[i], sizeof(int)); }
        c_test_stop = clock();
        clSVMFree(context, svm);
      }
      for (i = 0; i < 100; i++) { if (indata[i] != outdata[i]) { printf("\nNote: Memory corruption occured during transfer(s)"); break; }}
      diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
      printf("\nTest done, time: %f ms", diff);
      break;
    case 2:
      printf("\nKernel start overhead test:");
      printf("\nNote: this test is the same for OCL1.2 and 2.0");
      cl_kernel kernel = clCreateKernel(program, "CLRunner", &err);
      if (err != CL_SUCCESS) { printf("createkernel %i", err); }
      cl_event event;
      const size_t local = 1;
      const size_t global = 1;
      printf("\nTest 1: 100 Kernel Launches - Wait for completion");
      c_test_start = clock();
      for (int i = 0; i < 100; i++)
      {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
        clFinish(queue);
      }
      c_test_stop = clock();
      diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
      printf("\nTest 1 done, time: %f ms", diff);

      printf("\nTest 2: 100 Kernel Launches - Using command queue");
      c_test_start = clock();
      for (int i = 0; i < 100; i++)
      {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
      }
      clFinish(queue);
      c_test_stop = clock();
      diff = (((float)c_test_stop - (float)c_test_start) / CLOCKS_PER_SEC ) * 1000;
      printf("\nTest 2 done, time: %f ms", diff);
      break;
    case 3:
      printf("\nUnknown function - not implemented, very similar to test 1");
      break;
    default:
      printf("\nError, unknown input function");
      break;
  }
  clReleaseContext(context);
  clReleaseCommandQueue(queue);
  printf("\n");
}
