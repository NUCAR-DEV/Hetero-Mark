/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 *
 * KMeans clustering
 *
 */

#include <stdio.h>    /* for printf */
#include <stdint.h>  /* for uint64 definition */
#include <stdlib.h>  /* for exit() definition */
#include <time.h>    /* for clock_gettime */
#include <string.h>
#include <math.h>
#include <iostream>
#include <string>
#include <cassert>
#include "src/common/cl_util/cl_util.h"

#include "include/kmeans_cl12.h"

#define BILLION 1000000000L

using namespace std;

KMEANS::KMEANS() {
}

KMEANS::~KMEANS() {
  //Managed by benchmarks
  // Free_mem()
}

void KMEANS::CleanUpKernels() {
  checkOpenCLErrors(clReleaseKernel(kernel_s),
                    "Failed to release kernel kernel_s");

  checkOpenCLErrors(clReleaseKernel(kernel2),
                    "Failed to release kernel kernel2");
}

void KMEANS::SetInitialParameters(FilePackage parameters) {
  // ------------------------- command line options -----------------------//
  //int     opt;
  //extern char   *optarg;
  isBinaryFile = 0;
  threshold = 0.001;          // default value
  max_nclusters = 5;            // default value
  min_nclusters = 5;            // default value
  isRMSE = 0;
  isOutput = 0;
  nloops = 1;                 // default value

  char    line[1024];
  ssize_t ret;  // add return value for read

  float  *buf;
  npoints = 0;
  nfeatures = 0;

  best_nclusters = 0;

  int i, j;

filename = parameters.filename;
isBinaryFile = parameters.binary;
threshold = parameters.threshold;
max_nclusters = parameters.max_cl;
min_nclusters = parameters.min_cl;
isRMSE = parameters.RMSE;
isOutput = parameters.output;
nloops = parameters.nloops;

  // ============== I/O begin ==============//

  // io_timing = omp_get_wtime();
  if (isBinaryFile) {  // Binary file input
      int infile;
      if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
      }

      ret = read(infile, &npoints, sizeof(int));

      if (ret == -1) {
        fprintf(stderr, "Error: failed to read, info: %s.%d\n",
                __FILE__, __LINE__);
      }

      ret = read(infile, &nfeatures, sizeof(int));
      if (ret == -1) {
        fprintf(stderr, "Error: failed to read, info: %s.%d\n",
                __FILE__, __LINE__);
      }

      // allocate space for features[][] and read attributes of all objects
      // defined in header file
      buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
      feature    = (float**)malloc(npoints*          sizeof(float*));
      feature[0] = (float*) malloc(npoints*nfeatures*sizeof(float));

      // fixme: svm buffer
      for (i = 1; i < npoints; i++)
        feature[i] = feature[i-1] + nfeatures;

      ret = read(infile, buf, npoints*nfeatures*sizeof(float));

      if (ret == -1) {
        fprintf(stderr, "Error: failed to read, info: %s.%d\n",
                __FILE__, __LINE__);
      }

      close(infile);
    } else {
      FILE *infile;
      if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
      }

      while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0)
          npoints++;
      }

      rewind(infile);

      while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
          // ignore the id (first attribute): nfeatures = 1;
          while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
          break;
        }
      }

      // allocate space for features[] and read attributes of all objects
      buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
      feature     = (float**)malloc(npoints*          sizeof(float*));
      feature[0]  = (float*) malloc(npoints*nfeatures*sizeof(float));

      // fixme : svm buffer
      for (i = 1; i < npoints; i++)
        feature[i] = feature[i-1] + nfeatures;

      rewind(infile);

      i = 0;

      while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;

        for (j = 0; j < nfeatures; j++) {
          buf[i] = atof(strtok(NULL, " ,\t\n"));
          i++;
        }
      }

      fclose(infile);
    }

  // io_timing = omp_get_wtime() - io_timing;

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", npoints);
  printf("Number of features: %d\n", nfeatures);

  // error check for clusters
  if (npoints < min_nclusters) {
      printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
             min_nclusters, npoints);
      exit(0);
    }

  // now features holds 2-dimensional array of features //
  memcpy(feature[0], buf, npoints*nfeatures*sizeof(float));
  free(buf);
}

void KMEANS::CL_initialize() {
  runtime    = clRuntime::getInstance();
  // OpenCL objects get from clRuntime class
  platform   = runtime->getPlatformID();
  context    = runtime->getContext();
  device     = runtime->getDevice();
  cmd_queue  = runtime->getCmdQueue(0);
}

void KMEANS::CL_build_program() {
  cl_int err;
  // Helper to read kernel file
  file = clFile::getInstance();
  file->open("kmeans.cl");

  const char *source = file->getSourceChar();
  prog = clCreateProgramWithSource(context, 1,
                                   (const char **)&source, NULL, &err);
  checkOpenCLErrors(err, "Failed to create Program with source...\n");

  // Create program with OpenCL 2.0 support
  err = clBuildProgram(prog, 0, NULL, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");
}

void KMEANS::CL_create_kernels() {
  cl_int err;
  // Create kernels
  kernel_s = clCreateKernel(prog, "kmeans_kernel_c", &err);
  checkOpenCLErrors(err, "Failed to create kmeans_kernel_c");

  kernel2 = clCreateKernel(prog, "kmeans_swap", &err);
  checkOpenCLErrors(err, "Failed to create kernel kmeans_swap");
}

void KMEANS::Create_mem() {
  cl_int err;

  // Create buffers
  d_feature = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             npoints * nfeatures * sizeof(float),
                             NULL,
                             &err);
  checkOpenCLErrors(err, "clCreateBuffer d_feature failed");

  d_feature_swap = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  npoints * nfeatures * sizeof(float),
                                  NULL,
                                  &err);
  checkOpenCLErrors(err, "clCreateBuffer d_feature_swap failed");

  d_membership = clCreateBuffer(context,
                                CL_MEM_READ_WRITE,
                                npoints * sizeof(int),
                                NULL,
                                &err);
  checkOpenCLErrors(err, "clCreateBuffer d_membership failed");

  d_cluster = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             nclusters * nfeatures  * sizeof(float),
                             NULL,
                             &err);
  checkOpenCLErrors(err, "clCreateBuffer d_cluster failed");

  membership_OCL = (int*) malloc(npoints * sizeof(int));
}

void KMEANS::Swap_features() {
  cl_int err;

  // fixme
  err = clEnqueueWriteBuffer(cmd_queue,
                             d_feature,
                             1,
                             0,
                             npoints * nfeatures * sizeof(float),
                             feature[0],
                             0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer d_feature");

  clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
  clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
  clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &npoints);
  clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &nfeatures);

  size_t global_work     = (size_t) npoints;
  size_t local_work_size = BLOCK_SIZE;

  if (global_work % local_work_size != 0)
    global_work = (global_work / local_work_size + 1) * local_work_size;

  err = clEnqueueNDRangeKernel(cmd_queue,
                               kernel2,
                               1,
                               NULL,
                               &global_work,
                               &local_work_size,
                               0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel()");
}

void KMEANS::Free_mem() {
  clReleaseMemObject(d_feature);
  clReleaseMemObject(d_feature_swap);
  clReleaseMemObject(d_cluster);
  clReleaseMemObject(d_membership);

  free(membership_OCL);
}

void KMEANS::Kmeans_ocl() {
  int i, j;

  cl_int err;

  // Ke Wang adjustable local group size 2013/08/07 10:37:33
  // work group size is defined by RD_WG_SIZE_1 or
  // RD_WG_SIZE_1_0 2014/06/10 17:00:41
  size_t global_work     = (size_t) npoints;
  size_t local_work_size = BLOCK_SIZE2;

  if (global_work % local_work_size !=0)
    global_work = (global_work / local_work_size + 1) * local_work_size;

  // fixme: use svm
  err = clEnqueueWriteBuffer(cmd_queue,
                             d_cluster,
                             1,
                             0,
                             nclusters * nfeatures * sizeof(float),
                             clusters[0],
                             0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer d_cluster");

  int size = 0; int offset = 0;

  clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
  clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
  clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);
  clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &npoints);
  clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &nclusters);
  clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &nfeatures);
  clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
  clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

  err = clEnqueueNDRangeKernel(cmd_queue,
                               kernel_s,
                               1,
                               NULL,
                               &global_work,
                               &local_work_size,
                               0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel(kernel_s)");

  clFinish(cmd_queue);

  // fixme : use svm
  err = clEnqueueReadBuffer(cmd_queue,
                            d_membership,
                            1,
                            0,
                            npoints * sizeof(int),
                            membership_OCL,
                            0, 0, 0);
  checkOpenCLErrors(err, "ERROR: Memcopy Out");

  int delta_tmp = 0;
  for (i = 0; i < npoints; i++) {
    int cluster_id = membership_OCL[i];
    new_centers_len[cluster_id]++;
    if (membership_OCL[i] != membership[i]) {
      // update membership
      delta_tmp++;
      membership[i] = membership_OCL[i];
    }
    for (j = 0; j < nfeatures; j++) {
      new_centers[cluster_id][j] += feature[i][j];
    }
  }

  delta = (float) delta_tmp;
}

void KMEANS::Kmeans_clustering() {
  int i, j, n = 0;  // counters
  int loop = 0, temp;
  int initial_points = npoints;
  int c = 0;

  // nclusters should never be > npoints
  // that would guarantee a cluster without points
  if (nclusters > npoints) {
    nclusters = npoints;
  }

  // fixme : use svm
  // allocate space for and initialize returning variable clusters[]
  clusters    = (float**) malloc(nclusters *             sizeof(float*));
  clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (i = 1; i < nclusters; i++) {
    clusters[i] = clusters[i-1] + nfeatures;
  }

  // initialize the random clusters
  initial = (int *) malloc (npoints * sizeof(int));
  for (i = 0; i < npoints; i++) {
    initial[i] = i;
  }

  // fixme
  // randomly pick cluster centers
  for (i = 0; i < nclusters && initial_points >= 0; i++) {
    // n = (int)rand() % initial_points;

    for (j = 0; j < nfeatures; j++)
      clusters[i][j] = feature[initial[n]][j];  // remapped

      // swap the selected index to the end (not really necessary,
      // could just move the end up)
      temp                      = initial[n];
      initial[n]                = initial[initial_points-1];
      initial[initial_points-1] = temp;
      initial_points--;
      n++;
    }

  // initialize the membership to -1 for all
  // fixme: use svm
  for (i=0; i < npoints; i++) {
    membership[i] = -1;
  }

  // allocate space for and initialize new_centers_len and new_centers
  new_centers_len = (int*) calloc(nclusters, sizeof(int));

  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (i = 1; i < nclusters; i++)
    new_centers[i] = new_centers[i-1] + nfeatures;

  // iterate until convergence
  do {
    Kmeans_ocl();

      // replace old cluster centers with new_centers
      // CPU side of reduction
      for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++) {
          if (new_centers_len[i] > 0) {  // take average i.e. sum/n
            clusters[i][j] = new_centers[i][j] / new_centers_len[i];
          }
          new_centers[i][j] = 0.0;  // set back to 0
        }
        new_centers_len[i] = 0;    // set back to 0
      }

      c++;
  } while ((delta > threshold) && (loop++ < 500));  // makes sure loop ends
  printf("iterated %d times\n", c);

  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);

  // clusters is pointed to tmp_cluster_centres
  // return clusters;
  tmp_cluster_centres = clusters;
}


void KMEANS::Clustering() {
  cluster_centres = NULL;
  index = 0;    // number of iteration to reach the best RMSE

  // fixme
  membership = (int*) malloc(npoints * sizeof(int));

  CL_initialize();
  CL_build_program();
  CL_create_kernels();


  min_rmse_ref = FLT_MAX;

  int i;
  // int nclusters;  // number of clusters

  // sweep k from min to max_nclusters to find the best number of clusters
  for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
    // cannot have more clusters than points
    if (nclusters > npoints)
      break;

    // allocate device memory, invert data array
    Create_mem();
    Swap_features();

    // iterate nloops times for each number of clusters //
    for (i = 0; i < nloops; i++) {
      Kmeans_clustering();

      if (cluster_centres) {
        free(cluster_centres[0]);
        free(cluster_centres);
      }

      cluster_centres = tmp_cluster_centres;

      // find the number of clusters with the best RMSE //
      if (isRMSE) {
        RMS_err();

        if (rmse < min_rmse_ref) {
          min_rmse_ref = rmse;         // update reference min RMSE
          min_rmse = min_rmse_ref;     // update return min RMSE
          best_nclusters = nclusters;  // update optimum number of clusters
          index = i;  // update number of iteration to reach best RMSE
        }
      }
    }
    Free_mem();  // free device memory
  }
  free(membership);
}

// multi-dimensional spatial Euclid distance square
float KMEANS::euclid_dist_2(float *pt1,
                            float *pt2) {
  int i;
  float ans = 0.0;

  for (i = 0; i < nfeatures; i++)
    ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

  return(ans);
}

int KMEANS::find_nearest_point(float  *pt,           // [nfeatures]
                               float  **pts) {       // [npts][nfeatures]
  int index_local = 0, i;
  float max_dist = FLT_MAX;

  // find the cluster center id with min distance to pt
  for (i = 0; i < nclusters; i++) {
    float dist;
    dist = euclid_dist_2(pt, pts[i]);  // no need square root
    if (dist < max_dist) {
      max_dist = dist;
      index_local = i;
    }
  }
  return(index_local);
}

void KMEANS::RMS_err() {
  int    i;
  int   nearest_cluster_index;    // cluster center id with min distance to pt
  float  sum_euclid = 0.0;        // sum of Euclidean distance squares
  //float  ret;                     // return value

  // pass data pointers
  float **feature_loc, **cluster_centres_loc;
  feature_loc = feature;
  cluster_centres_loc = tmp_cluster_centres;

  // calculate and sum the sqaure of euclidean distance
/* #pragma omp parallel for                      \
  shared(feature_loc, cluster_centres_loc)      \
  firstprivate(npoints, nfeatures, nclusters)   \
  private(i, nearest_cluster_index)             \
  schedule(static)
*/

  for (i = 0; i < npoints; i++) {
      nearest_cluster_index = find_nearest_point(feature_loc[i],
                                                 cluster_centres_loc);
      sum_euclid += euclid_dist_2(feature_loc[i],
                                  cluster_centres_loc[nearest_cluster_index]);
  }

  // divide by n, then take sqrt
  rmse = sqrt(sum_euclid / npoints);
}

void KMEANS::Display_results() {
  int i, j;

  // cluster center coordinates : displayed only for when k=1
  if ((min_nclusters == max_nclusters) && (isOutput == 1)) {
      printf("\n================= Centroid Coordinates =================\n");
      for (i = 0; i < max_nclusters; i++) {
        printf("%d:", i);
        for (j = 0; j < nfeatures; j++) {
          printf(" %.2f", cluster_centres[i][j]);
        }
        printf("\n\n");
      }
    }

  // float len = (float) ((max_nclusters - min_nclusters + 1)*nloops);
  printf("Number of Iteration: %d\n", nloops);
  // printf("Time for I/O: %.5fsec\n", io_timing);
  // printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

  if (min_nclusters != max_nclusters) {
      if (nloops != 1) {
        // range of k, multiple iteration
        // printf("Average Clustering Time: %fsec\n",
        //      cluster_timing / len);
        printf("Best number of clusters is %d\n", best_nclusters);
      } else {
        // range of k, single iteration
        // printf("Average Clustering Time: %fsec\n",
        //      cluster_timing / len);
        printf("Best number of clusters is %d\n", best_nclusters);
      }
    } else {
    if (nloops != 1) {
      // single k, multiple iteration
      // printf("Average Clustering Time: %.5fsec\n",
      //      cluster_timing / nloops);
      if (isRMSE) {  // if calculated RMSE
        printf("Number of trials to approach the best RMSE of %.3f is %d\n",
               min_rmse, index + 1);
      }
    } else {
      // single k, single iteration
      if (isRMSE) {
        // if calculated RMSE
        printf("Root Mean Squared Error: %.3f\n", min_rmse);
      }
    }
  }
}

void KMEANS::Run() {
  // ----------------- Read input file and allocate features --------------//
  //Read already started...

  // ----------------- Clustering -------------------------- --------------//
  // cluster_timing = omp_get_wtime();  // Total clustering time
  Clustering();
  // cluster_timing = omp_get_wtime() - cluster_timing;

  Display_results();
}
