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

KmeansBenchmark::KmeansBenchmark() {}

KmeansBenchmark::~KmeansBenchmark() {}

void KmeansBenchmark::Read() {
  FILE *input_file;
  if ((input_file = fopen(filename_.c_str(), "r")) == NULL) {
    fprintf(stderr, "Error: no such file (%s)\n", filename_.c_str());
    exit(1);
  }

  num_points_ = GetNumberPointsFromInputFile(input_file);
  num_features_ = GetNumberFeaturesFromInputFile(input_file);

  feature_ = new float *[num_points_];
  feature_[0] = new float[num_points_ * num_features_];

  for (int i = 1; i < num_points_; i++) {
    feature_[i] = feature_[i - 1] + num_features_;
  }

  LoadFeaturesFromInputFile(input_file);

  fclose(input_file);

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", num_points_);
  printf("Number of features: %d\n", num_features_);

  // error check for clusters
  if (num_points_ < min_clusters_) {
    printf("Error: min_clusters_(%d) > num_points_(%d) -- cannot proceed\n",
           min_clusters_, num_points_);
    exit(0);
  }
}

int KmeansBenchmark::GetNumberPointsFromInputFile(FILE *input_file) {
  char line[1024];
  char *save_line;
  int num_points = 0;

  rewind(input_file);
  while (fgets(line, sizeof(line), input_file) != NULL) {
    if (strtok_r(line, " \t\n", &save_line) != 0) {
      num_points++;
    }
  }

  return num_points;
}

int KmeansBenchmark::GetNumberFeaturesFromInputFile(FILE *input_file) {
  char line[1024];
  char *save_line;
  int num_features = 0;

  rewind(input_file);
  while (fgets(line, sizeof(line), input_file) != NULL) {
    if (strtok_r(line, " \t\n", &save_line) == NULL) continue;

    while (strtok_r(NULL, " ,\t\n", &save_line) != NULL) {
      num_features++;
    }
    break;
  }

  return num_features;
}

void KmeansBenchmark::LoadFeaturesFromInputFile(FILE *input_file) {
  char line[1024];
  char *save_line;

  int index = 0;
  rewind(input_file);
  while (fgets(line, 1024, input_file) != NULL) {
    // This if can avoid the index of the point to be insert into the
    // feature array
    if (strtok_r(line, " \t\n", &save_line) == NULL) continue;

    for (int j = 0; j < num_features_; j++) {
      feature_[0][index] = atof(strtok_r(NULL, " ,\t\n", &save_line));
      index++;
    }
  }
}

void KmeansBenchmark::CreateTemporaryMemory() {
  feature_transpose_ = new float[num_points_ * num_features_];
}

void KmeansBenchmark::TransposeFeatures() {
  size_t global_work = (size_t)num_points_;
  size_t local_work_size = BLOCK_SIZE;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;
  kmeans_swap(feature_[0], feature_transpose_, num_points_, num_features_,
              lparm);
}

void KmeansBenchmark::Free_mem() { delete feature_transpose_; }

void KmeansBenchmark::UpdateMembership(int num_clusters) {
  int *new_membership = new int[num_points_];
  size_t global_work = (size_t)num_points_;
  size_t local_work_size = BLOCK_SIZE;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;

  int size = 0;
  int offset = 0;
  kmeans_kernel_c(feature_transpose_, clusters_[0], new_membership, num_points_,
                  num_clusters, num_features_, offset, size, lparm);

  delta = 0;
  for (int i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta++;
      membership_[i] = new_membership[i];
    }
  }
}

void KmeansBenchmark::DumpMembership() {
  for (int i = 0; i < num_points_; i++) {
    printf("Point %d, belongs to cluster %d\n", i, membership_[i]);
  }
}

void KmeansBenchmark::KmeansClustering(int num_clusters) {
  int num_iteration = 0;

  // that would guarantee a cluster without points
  if (num_clusters > num_points_) {
    printf("Number of clusters cannot be less than the number of points\n");
    exit(1);
  }

  InitializeClusters(num_clusters);
  InitializeMembership();

  // iterate until convergence
  do {
    UpdateMembership(num_clusters);
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta > 0) && (num_iteration < 500));

  printf("iterated %d times\n", num_iteration);
}

void KmeansBenchmark::UpdateClusterCentroids(int num_clusters) {
  // Allocate space for and initialize new_centers_len and new_centers
  int *member_count = new int[num_clusters]();

  // Clean up clusters_
  for (int i = 0; i < num_clusters; i++) {
    for (int j = 0; j < num_features_; j++) {
      clusters_[i][j] = 0;
    }
  }

  // Calculate sum
  for (int i = 0; i < num_points_; i++) {
    for (int j = 0; j < num_features_; j++) {
      clusters_[membership_[i]][j] += feature_[i][j];
    }
    member_count[membership_[i]]++;
  }

  // For each cluster, devide by the number of points in the cluster
  for (int i = 0; i < num_clusters; i++) {
    for (int j = 0; j < num_features_; j++) {
      clusters_[i][j] /= member_count[i];
    }
  }
}

void KmeansBenchmark::InitializeClusters(int num_clusters) {
  clusters_ = new float *[num_clusters];
  clusters_[0] = new float[num_clusters * num_features_];
  for (int i = 1; i < num_clusters; i++) {
    clusters_[i] = clusters_[i - 1] + num_features_;
  }

  for (int i = 0; i < num_clusters; i++) {
    for (int j = 0; j < num_features_; j++) {
      clusters_[i][j] = feature_[i][j];
    }
  }
}

void KmeansBenchmark::InitializeMembership() {
  for (int i = 0; i < num_points_; i++) {
    membership_[i] = -1;
  }
}

void KmeansBenchmark::DumpClusterCentroids(int num_clusters) {
  for (int i = 0; i < num_clusters; i++) {
    printf("Clusters %d: ", i);
    for (int j = 0; j < num_features_; j++) {
      printf("%3.2f, ", clusters_[i][j]);
    }
    printf("\n");
  }
}

void KmeansBenchmark::Clustering() {
  membership_ = new int[num_points_];
  min_rmse_ = FLT_MAX;

  // sweep k from min to max_clusters_ to find the best number of clusters
  for (int num_clusters = min_clusters_; num_clusters <= max_clusters_;
       num_clusters++) {
    // cannot have more clusters than points
    if (num_clusters > num_points_) break;

    CreateTemporaryMemory();
    TransposeFeatures();

    KmeansClustering(num_clusters);

    float rmse = CalculateRMSE();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters;
    }
  }

  delete[] membership_;
}

float KmeansBenchmark::CalculateRMSE() {
  float mean_square_error = 0;
  for (int i = 0; i < num_points_; i++) {
    float distance_square = 0;
    for (int j = 0; j < num_features_; j++) {
      distance_square +=
          pow((feature_[i][j] - clusters_[membership_[i]][j]), 2);
    }
    mean_square_error += distance_square;
  }
  mean_square_error /= num_points_;
  return sqrt(mean_square_error);
}

void KmeansBenchmark::DisplayResults() {
  int i, j;

  // cluster center coordinates : displayed only for when k=1
  if (min_clusters_ == max_clusters_) {
    printf("\n================= Centroid Coordinates =================\n");
    for (i = 0; i < max_clusters_; i++) {
      printf("%d:", i);
      for (j = 0; j < num_features_; j++) {
        printf(" %.2f", clusters_[i][j]);
      }
      printf("\n\n");
    }
  }

  if (min_clusters_ != max_clusters_) {
    // range of k, single iteration
    // printf("Average Clustering Time: %fsec\n",
    //      cluster_timing / len);
    printf("Best number of clusters is %d\n", best_num_clusters_);
  }
  printf("Root Mean Squared Error: %.3f\n", min_rmse_);
}

void KmeansBenchmark::Initialize() {
  timer_->End({"Initialize"});
  timer_->Start();
  kmeans_kernel_c_init(0);
  kmeans_swap_init(0);
  timer_->End({"Init Runtime"});
  timer_->Start();
  Read();
}

void KmeansBenchmark::Run() { Clustering(); }

void KmeansBenchmark::Summarize() { DisplayResults(); }

void KmeansBenchmark::Verify() {}

void KmeansBenchmark::Cleanup() { Free_mem(); }
