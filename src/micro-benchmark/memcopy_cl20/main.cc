#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>

#define BILLION 1000000000L

int main(int argc, const char * argv[]) {
  uint64_t diff;

  struct timespec start, end;

  FILE *cl_code = fopen("kernel.cl", "r");

  if (cl_code == NULL) {
    printf("\nerror: clfile\n");
    return(1);
  }

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

  if (err != CL_SUCCESS) {
    printf("platformid %i", err);
    return 1;
  }


  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  if (err != CL_SUCCESS) {
    printf("deviceid %i", err);
    return 1;
  }

  context = clCreateContext(0, 1, &device, NULL, NULL, &err);

  if (err != CL_SUCCESS) {
    printf("createcontext %i", err);
    return 1;
  }

  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

  if (err != CL_SUCCESS) {
    printf("commandqueue %i", err);
    return 1;
  }

  program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_length, &err);

  if (err != CL_SUCCESS) {
    printf("createprogram %i", err);
    return 1;
  }


  err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL2.0", NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("buildprogram ocl20 %i", err);
  }


  if (err == CL_BUILD_PROGRAM_FAILURE) {

    size_t log_size;

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    char *log = (char *) malloc(log_size);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    printf("%s\n", log);

    return 1;

  }


  int i;


  printf("\nmemtime int copy test:");

  for (int x = 1; x < 11; x++)
  {
    uint64_t tempdiff = 0;
    // Repeat the experiment 1000 times and average it
    for (int y = 1; y <= 1000; y++) {

//      printf("\nTest %i, %i objects", x, x*10000);

      int *indata = (int *)malloc(sizeof(int)*x*10000);

      int *outdata = (int *)malloc(sizeof(int)*x*10000);

      for (i = 0; i < x*10000; i++) {
        indata[i] = rand();
      }

      clock_gettime(CLOCK_MONOTONIC, &start);

      int *svm = (int *)clSVMAlloc(context, CL_MEM_SVM_FINE_GRAIN_BUFFER,
                                   sizeof(int)*x*10000, 0);

/*      err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm,
        sizeof(int)*x*10000, 0, 0, 0);

      if (err != CL_SUCCESS) {
        printf("enqueuesvmmap ocl20 %i", err);
      }
*/

      for (i = 0; i < x*10000; i++) {
        memcpy(&svm[i], &indata[i], sizeof(int));
      }

/*    err = clEnqueueSVMUnmap(queue, svm, 0, 0, 0);

      if (err != CL_SUCCESS) {
        printf("enqueueunmap ocl20 %i", err);
      }

      err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ,
      svm, sizeof(int)*x*10000, 0, 0, 0);

      if (err != CL_SUCCESS) {
        printf("enqueusvmmap2 ocl20 %i", err);
      }
*/
      for (i = 0; i < x*10000; i++) {
        memcpy(&outdata[i], &svm[i], sizeof(int));
      }

      clock_gettime(CLOCK_MONOTONIC, &end);

      clSVMFree(context, svm);

      for (i = 0; i < x*10000; i++) {
        if (indata[i] != outdata[i]) {
          printf("\nNote: Memory corruption occured during transfer(s)");
          break;
        }
      }
      diff = BILLION * (end.tv_sec - start.tv_sec)
        + end.tv_nsec - start.tv_nsec;
      tempdiff = tempdiff + diff;
    }

    printf("\n Test %i done, time: %llu nanoseconds\n", x,
           (long long unsigned int) tempdiff/1000);
  }

  clReleaseContext(context);

  clReleaseCommandQueue(queue);

  printf("\n");
}
