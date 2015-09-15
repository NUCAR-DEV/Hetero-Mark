/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*
 * Mofified by: Leiming Yu (ylm@ece.neu.edu)
 * Mofified by: Yifan Sun (yifansun@coe.neu.edu)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <string>
#include <cassert>
#include <memory>

#include "src/hsa/kmeans_hsa/kmeans_benchmark.h"

KmeansBenchmark::KmeansBenchmark() {
}

KmeansBenchmark::~KmeansBenchmark() {}

void KmeansBenchmark::Read() {
  char line[1024];
  char *save_line;
  float *buf;
  FILE *infile;
  if ((infile = fopen(filename_.c_str(), "r")) == NULL) {
    fprintf(stderr, "Error: no such file (%s)\n", filename_.c_str());
    exit(1);
  }

  while (fgets(line, 1024, infile) != NULL) {
    if (strtok_r(line, " \t\n", &save_line) != 0) npoints++;
  }

  rewind(infile);

  while (fgets(line, 1024, infile) != NULL) {
    if (strtok_r(line, " \t\n", &save_line) != 0) {
      // ignore the id (first attribute): nfeatures = 1;
      while (strtok_r(NULL, " ,\t\n", &save_line) != NULL) nfeatures++;
      break;
    }
  }

  // allocate space for features[] and read attributes of all objects
  buf = new float[npoints * nfeatures];
  feature = new float*[npoints];
  feature[0] = new float[npoints * nfeatures];

  // fixme : svm buffer
  for (int i = 1; i < npoints; i++) feature[i] = feature[i - 1] + nfeatures;

  rewind(infile);

  int i = 0;
  while (fgets(line, 1024, infile) != NULL) {
    if (strtok_r(line, " \t\n", &save_line) == NULL) continue;

    for (int j = 0; j < nfeatures; j++) {
      buf[i] = atof(strtok_r(NULL, " ,\t\n", &save_line));
      i++;
    }
  }

  fclose(infile);

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
  memcpy(feature[0], buf, npoints * nfeatures * sizeof(float));
  free(buf);
}

void KmeansBenchmark::Create_mem() {
  // Create buffers
  d_feature = new float[npoints * nfeatures];
  d_feature_swap = new float[npoints * nfeatures];
  d_membership = new int[npoints];
  d_cluster = new float[nclusters * nfeatures];
  membership_OCL = new int[npoints];
}

void KmeansBenchmark::Swap_features() {
  size_t global_work = (size_t)npoints;
  size_t local_work_size = BLOCK_SIZE;
  if (global_work % local_work_size != 0)
    global_work = (global_work / local_work_size + 1) * local_work_size;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;
  kmeans_swap(feature[0], d_feature_swap, npoints, nfeatures, lparm);
}

void KmeansBenchmark::Free_mem() {
  /*
  delete[] d_feature;
  delete[] d_feature_swap;
  delete[] d_cluster;
  delete[] d_membership;
  delete[] membership_OCL;
  */
}

void KmeansBenchmark::Kmeans_ocl() {
  printf("iterating\n");
  int i, j;

  // Ke Wang adjustable local group size 2013/08/07 10:37:33
  // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10
  // 17:00:41
  size_t global_work = (size_t)npoints;
  size_t local_work_size = BLOCK_SIZE2;

  if (global_work % local_work_size != 0)
    global_work = (global_work / local_work_size + 1) * local_work_size;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;

  int size = 0;
  int offset = 0;
  kmeans_kernel_c(d_feature_swap, clusters[0], membership_OCL, npoints,
                  nclusters, nfeatures, offset, size, lparm);

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

  delta = static_cast<float>(delta_tmp);
}

void KmeansBenchmark::Kmeans_clustering() {
  printf("Running\n");
  int i, j, n = 0;  // counters
  int loop = 0, temp;
  int initial_points = npoints;
  int c = 0;

  // nclusters should never be > npoints
  // that would guarantee a cluster without points
  if (nclusters > npoints) {
    nclusters = npoints;
  }

  // allocate space for and initialize returning variable clusters[]
  clusters = new float*[nclusters];
  clusters[0] = new float[nclusters * nfeatures];
  for (i = 1; i < nclusters; i++) {
    clusters[i] = clusters[i - 1] + nfeatures;
  }

  // initialize the random clusters
  initial = new int[npoints];
  for (i = 0; i < npoints; i++) {
    initial[i] = i;
  }

  // randomly pick cluster centers
  for (i = 0; i < nclusters && initial_points >= 0; i++) {
    // n = (int)rand() % initial_points;

    for (j = 0; j < nfeatures; j++)
      clusters[i][j] = feature[initial[n]][j];  // remapped

    // swap the selected index to the end (not really necessary,
    // could just move the end up)
    temp = initial[n];
    initial[n] = initial[initial_points - 1];
    initial[initial_points - 1] = temp;
    initial_points--;
    n++;
  }

  // initialize the membership to -1 for all
  for (i = 0; i < npoints; i++) {
    membership[i] = -1;
  }

  // allocate space for and initialize new_centers_len and new_centers
  new_centers_len = new int[nclusters]();
  new_centers = new float*[nclusters];
  new_centers[0] = new float[nclusters * nfeatures]();

  for (i = 1; i < nclusters; i++)
    new_centers[i] = new_centers[i - 1] + nfeatures;

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
      new_centers_len[i] = 0;  // set back to 0
    }

    c++;
  } while ((delta > threshold) && (loop++ < 500));

  printf("iterated %d times\n", c);

  delete[] new_centers[0];
  delete[] new_centers;
  delete[] new_centers_len;

  // clusters is pointed to tmp_cluster_centres
  // return clusters;
  tmp_cluster_centres = clusters;
}

void KmeansBenchmark::Clustering() {
  cluster_centres = NULL;
  index = 0;  // number of iteration to reach the best RMSE

  // fixme
  membership = new int[npoints];

  min_rmse_ref = FLT_MAX;

  int i;

  // sweep k from min to max_nclusters to find the best number of clusters
  for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
    // cannot have more clusters than points
    if (nclusters > npoints) break;

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
          min_rmse_ref = rmse;  // update reference min RMSE
          min_rmse = min_rmse_ref;  // update return min RMSE
          best_nclusters = nclusters;  // update optimum number of clusters
          index = i;  // update number of iteration to reach best RMSE
        }
      }
    }
  }

  delete[] membership;
}

// multi-dimensional spatial Euclid distance square
float KmeansBenchmark::euclid_dist_2(float *pt1, float *pt2) {
  int i;
  float ans = 0.0;

  for (i = 0; i < nfeatures; i++) ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

  return (ans);
}

int KmeansBenchmark::find_nearest_point(float *pt, float **pts) {
  int index_local, i;
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
  return (index_local);
}

void KmeansBenchmark::RMS_err() {
  int i;
  int nearest_cluster_index;  // cluster center id with min distance to pt
  float sum_euclid = 0.0;     // sum of Euclidean distance squares

  // pass data pointers
  float **feature_loc, **cluster_centres_loc;
  feature_loc = feature;
  cluster_centres_loc = tmp_cluster_centres;

  // calculate and sum the sqaure of euclidean distance
  for (i = 0; i < npoints; i++) {
    nearest_cluster_index =
        find_nearest_point(feature_loc[i], cluster_centres_loc);

    sum_euclid += euclid_dist_2(feature_loc[i],
                                cluster_centres_loc[nearest_cluster_index]);
  }

  // divide by n, then take sqrt
  rmse = sqrt(sum_euclid / npoints);
}

void KmeansBenchmark::Display_results() {
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

  // float len = (float)((max_nclusters - min_nclusters + 1) * nloops);

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

void KmeansBenchmark::Initialize() {
  Read();
}

void KmeansBenchmark::Run() {
  Clustering();
}

void KmeansBenchmark::Summarize() {
  Display_results();
}

void KmeansBenchmark::Verify() {
}

void KmeansBenchmark::Cleanup() {
    Free_mem();  // free device memory
}
