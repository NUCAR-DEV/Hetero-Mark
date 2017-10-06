/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Yifan Sun (yifansun@coe.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include "src/knn/cuda/knn_cuda_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void knn_cuda(LatLong *latLong,float *d_distances,int num_records,float lat,float lng) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_records) {
	  d_distances[tid] = (float)sqrt((lat - latLong[tid].lat)*(lat-latLong[tid].lat)+(lng-latLong[tid].lng)*(lng-latLong[tid].lng));
   }
}

void KnnCudaBenchmark::Initialize() {
 KnnBenchmark::Initialize();
 printf("Block size is %d \n", num_records_);
 h_distances_ = new float[num_records_];
 cudaMalloc((void **)&d_distances_,sizeof(float) * num_records_);
 cudaMalloc((void **)&d_locations_,sizeof(LatLong) * num_records_);
 cudaMemcpy(d_locations_,&locations_[0],sizeof(LatLong)*num_records_,cudaMemcpyHostToDevice);
}

void KnnCudaBenchmark::Run() {
 dim3 block_size(64);
 dim3 grid_size((num_records_ + 64 - 1) / 64 );
 printf("Grid size is %d \n", grid_size.x);
 knn_cuda<<<grid_size, block_size>>>(d_locations_,d_distances_,num_records_,latitude_,longitude_);
 cudaDeviceSynchronize();
 
 cudaMemcpy(h_distances_, d_distances_, sizeof(float) * num_records_, cudaMemcpyDeviceToHost);

 for(int i = 0; i < 10; i++)
        printf("Distances are %f \n", h_distances_[i]);
  
 // find the resultsCount least distances
 findLowest(records_,h_distances_,num_records_,k_value_);

 for(int i = 0;i < k_value_;i++) {
      printf("%s --> Distance=%f\n",records_[i].recString,records_[i].distance);
    }
}

void KnnCudaBenchmark::Cleanup() {
  cudaFree(d_distances_);
  cudaFree(d_locations_);
  free(output_distances_);
  KnnBenchmark::Cleanup();
}
