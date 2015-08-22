#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>
#include "src/hsa/fir_cl20/kernels.h"

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

unsigned int numTap = 0;
unsigned int numData = 0;		// Block size
unsigned int numTotalData = 0;
unsigned int numBlocks = 0;		// Number of blocks
float* input = NULL;
float* output = NULL;
float* coeff = NULL;
float* temp_output = NULL;

int main(int argc , char** argv) {

  uint64_t diff;
  struct timespec start, end;
  
  /* measure monotonic time */
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

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

  // Allocate SVM buffers
  input = (float *)malloc(numTotalData * sizeof(float));
  output = (float *)malloc(numTotalData * sizeof(float));
  coeff = (float *)malloc(numTap * sizeof(float));
  temp_output = (float *)malloc((numData + numTap - 1) * sizeof(float));
    
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
    

  // Decide the local group size formation
  size_t globalThreads[1]={numData};
  size_t localThreads[1]={128};
  count = 0;

  // FIR Loop
  uint64_t execTimeMs = 0.0;
  while( count < numBlocks )
    {	
      // Custom item size based on current algorithm
      size_t global_item_size = numData; 
      size_t local_item_size = numData;

      // Execute the OpenCL kernel on the list
      SNK_INIT_LPARM(lparm, 0);
      lparm->ndim = 1;
      lparm->gdims[0] = globalThreads[0];
      lparm->ldims[0] = localThreads[0];
      FIR(output, coeff, temp_output, numTap, lparm);

      /* Kernel Profiling */
      uint64_t kernel_diff;
      struct timespec kernel_start, kernel_end;

      /* measure monotonic time */
      clock_gettime(CLOCK_MONOTONIC, &kernel_start);/* mark start time */
      
      clock_gettime(CLOCK_MONOTONIC, &kernel_end);/* mark the end time */
      
      uint64_t tmp = BILLION * (kernel_end.tv_sec - kernel_start.tv_sec) + kernel_end.tv_nsec - kernel_start.tv_nsec;
      execTimeMs += tmp;

      count ++;
    }
  printf("\nKernel exec time: %llu nanoseconds\n", (long long unsigned int)execTimeMs);

  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

  return 0;
}
