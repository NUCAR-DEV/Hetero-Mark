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

#include "src/kmeans/kmeans_benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

void KmeansBenchmark::Initialize() {
  std::vector<float> points;
  if (num_points_ != 0) {
    for (unsigned i = 0; i < num_points_; i++) {
      for (unsigned j = 0; j < num_features_; j++) {
        points.push_back(rand());
      }
    }
  } else {
    std::ifstream file(filename_);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file" << std::endl;
      exit(-1);
    }
    std::string line;

    // Read points from input file
    while (std::getline(file, line)) {
      num_points_++;

      std::stringstream ss(line);
      float n;

      // Ignore the 1st attribute
      bool ignore = true;
      ss >> n;
      while (ss.good()) {
        if (!ignore) points.push_back(n);
        ignore = false;
        ss >> n;
      }
    }

    // Count number features per point
    num_features_ = points.size() / num_points_;

    // Sanity check
    if (num_points_ < min_num_clusters_)
      std::cerr << "Error: More clusters than points" << std::endl;
  }

  // Copy data to host buffer
  host_features_ = new float[num_points_ * num_features_];
  std::memcpy(host_features_, points.data(),
              num_points_ * num_features_ * sizeof(float));
}

void KmeansBenchmark::DumpFeatures() {
  for (uint32_t i = 0; i < num_points_; i++) {
    printf("Point[%d]: ", i);
    for (uint32_t j = 0; j < num_features_; j++) {
      printf("%.0f, ", host_features_[i * num_features_ + j]);
    }
    printf("\n");
  }
}

void KmeansBenchmark::Verify() {
  membership_ = new int[num_points_];
  cpu_min_rmse_ = FLT_MAX;

  for (uint32_t num_clusters = min_num_clusters_;
       num_clusters <= max_num_clusters_; num_clusters++) {
    if (num_clusters > num_points_) break;
    KmeansClusteringCpu(num_clusters);

    float rmse = CalculateRMSE();
    if (rmse < cpu_min_rmse_) {
      cpu_min_rmse_ = rmse;
      best_num_clusters_ = num_clusters;
    }
  }

  delete[] membership_;

  if (std::abs(cpu_min_rmse_ - min_rmse_) < cpu_min_rmse_ * 0.001) {
    printf("Passed! (%f) (%f)\n", min_rmse_, cpu_min_rmse_);
  } else {
    printf("Failed! Expected to be %f, but got %f\n", cpu_min_rmse_, min_rmse_);
    exit(-1);
  }
}

void KmeansBenchmark::DumpMembership() {
  for (unsigned i = 0; i < num_points_; i++) {
    printf("Point %d, belongs to cluster %d\n", i, membership_[i]);
  }
}

void KmeansBenchmark::DumpClusterCentroids(unsigned num_clusters) {
  for (uint32_t i = 0; i < num_clusters; i++) {
    printf("Centroid[%d]: ", i);
    for (uint32_t j = 0; j < num_features_; j++) {
      printf("%.2f, ", clusters_[i * num_features_ + j]);
    }
    printf("\n");
  }
}

void KmeansBenchmark::KmeansClusteringCpu(uint32_t num_clusters) {
  int num_iteration = 0;
  InitializeClusters(num_clusters);
  InitializeMembership();

  do {
    UpdateMembershipCpu(num_clusters);
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < num_loops_));

  printf("cpu iterated %d times\n", num_iteration);
}

void KmeansBenchmark::UpdateMembershipCpu(unsigned num_clusters) {
  int *new_membership = new int[num_points_];

  for (uint32_t i = 0; i < num_points_; i++) {
    float min_dist = FLT_MAX;
    int index = 0;
    for (uint32_t j = 0; j < num_clusters; j++) {
      float dist = 0;
      for (uint32_t k = 0; k < num_features_; k++) {
        dist += pow(host_features_[i * num_features_ + k] -
                        clusters_[j * num_features_ + k],
                    2);
      }

      if (dist < min_dist) {
        min_dist = dist;
        index = j;
      }
    }
    new_membership[i] = index;
  }

  delta_ = 0;
  for (uint32_t i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }

  delete[] new_membership;
}

void KmeansBenchmark::UpdateClusterCentroids(unsigned num_clusters) {
  cpu_gpu_logger_->CPUOn();
  // Allocate space for and initialize new_centers_len and new_centers
  int *member_count = new int[num_clusters]();

  // Clean up clusters_
  for (unsigned i = 0; i < num_clusters * num_features_; i++) {
    clusters_[i] = 0;
  }

  // Calculate sum
  for (unsigned i = 0; i < num_points_; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = membership_[i] * num_features_ + j;
      clusters_[index_cluster] += host_features_[index_feature];
    }
    member_count[membership_[i]]++;
  }
  // For each cluster, divide by the number of points in the cluster
  for (unsigned i = 0; i < num_clusters; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index = i * num_features_ + j;
      if (member_count[i]) clusters_[index] /= member_count[i];
    }
  }

  delete[] member_count;
  cpu_gpu_logger_->CPUOff();
}

void KmeansBenchmark::InitializeClusters(unsigned num_clusters) {
  clusters_ = new float[num_clusters * num_features_];
  // for (unsigned i = 1; i < num_clusters; i++)
  //   clusters_[i] = clusters_[i - 1] + num_features_;
  //
  for (unsigned i = 0; i < num_clusters * num_features_; i++)
    clusters_[i] = host_features_[i];
}

void KmeansBenchmark::InitializeMembership() {
  for (unsigned i = 0; i < num_points_; i++) membership_[i] = -1;
}

float KmeansBenchmark::CalculateRMSE() {
  cpu_gpu_logger_->CPUOn();
  float mean_square_error = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    float distance_square = 0;
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = membership_[i] * num_features_ + j;
      distance_square +=
          pow((host_features_[index_feature] - clusters_[index_cluster]), 2);
    }
    mean_square_error += distance_square;
  }
  mean_square_error /= num_points_;
  mean_square_error = sqrt(mean_square_error);
  cpu_gpu_logger_->CPUOff();
  return mean_square_error;
}

void KmeansBenchmark::Summarize() {}

void KmeansBenchmark::Cleanup() {
  delete[] host_features_;
  delete[] clusters_;
}
