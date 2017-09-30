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

#include <cmath>
#include <fstream>
#include <string>

#include "src/fdeb/fdeb_benchmark.h"

void FdebBenchmark::Initialize() {
  std::string value_str;
  float value;
  std::ifstream file(input_file_);

  while (file.good()) {
    getline(file, value_str, ',');
    if (value_str == "") break;
    std::cout << value_str << ",";
    value = std::stof(value_str);
    edge_src_x_.push_back(value);

    getline(file, value_str, ',');
    if (value_str == "") break;
    std::cout << value_str << ",";
    value = std::stof(value_str);
    edge_src_y_.push_back(value);

    getline(file, value_str, ',');
    if (value_str == "") break;
    std::cout << value_str << ",";
    value = std::stof(value_str);
    edge_dst_x_.push_back(value);

    getline(file, value_str);
    if (value_str == "") break;
    std::cout << value_str << "\n";
    value = std::stof(value_str);
    edge_dst_y_.push_back(value);

    edge_count_++;
  }
}

void FdebBenchmark::PrintEdges() {
  for (int i = 0; i < edge_count_; i++) {
    printf("[%d] (%f, %f) -> (%f, %f)\n", i, edge_src_x_[i], edge_src_y_[i],
           edge_dst_x_[i], edge_dst_y_[i]);
  }
}

void FdebBenchmark::Verify() { FdebCpu(); }

void FdebBenchmark::FdebCpu() {
  CalculateCompatibility();
  BundlingCpu();
  SaveSubdevisedEdges("out.data");
}

void FdebBenchmark::CalculateCompatibility() {
  compatibility_.resize(edge_count_);
  for (int i = 0; i < edge_count_; i++) {
    compatibility_[i].resize(edge_count_);
    for (int j = 0; j < edge_count_; j++) {
      if (i == j) {
        compatibility_[i][j] = 1;
        continue;
      }

      // Fill in the real algorithm here.
      compatibility_[i][j] =
          1.0 * AngleCompatibility(i, j) * ScaleCompatibility(i, j) *
          PositionCompatibility(i, j) * VisibilityCompatibility(i, j);
      printf("compatibility [%d, %d] %f\n", i, j, compatibility_[i][j]);
    }
  }
}

float FdebBenchmark::AngleCompatibility(int i, int j) {
  float e1x1 = edge_src_x_[i];
  float e1y1 = edge_src_y_[i];
  float e1x2 = edge_dst_x_[i];
  float e1y2 = edge_dst_y_[i];
  float e2x1 = edge_src_x_[j];
  float e2y1 = edge_src_y_[j];
  float e2x2 = edge_dst_x_[j];
  float e2y2 = edge_dst_y_[j];
  float e1x = e1x2 - e1x1;
  float e1y = e1y2 - e1y1;
  float e2x = e2x2 - e2x1;
  float e2y = e2y2 - e2y1;
  float l1 = sqrt(e1x * e1x + e1y * e1y);
  float l2 = sqrt(e2x * e2x + e2y * e2y);

  float inner = e1x * e2x + e1y * e2y;
  float length_product = l1 * l2;
  if (length_product == 0) {
    return 1;
  }
  return std::abs(inner / length_product);
}


float FdebBenchmark::ScaleCompatibility(int i, int j) { 
  float e1x1 = edge_src_x_[i];
  float e1y1 = edge_src_y_[i];
  float e1x2 = edge_dst_x_[i];
  float e1y2 = edge_dst_y_[i];
  float e2x1 = edge_src_x_[j];
  float e2y1 = edge_src_y_[j];
  float e2x2 = edge_dst_x_[j];
  float e2y2 = edge_dst_y_[j];
  float e1x = e1x2 - e1x1;
  float e1y = e1y2 - e1y1;
  float e2x = e2x2 - e2x1;
  float e2y = e2y2 - e2y1;
  float l1 = sqrt(e1x * e1x + e1y * e1y);
  float l2 = sqrt(e2x * e2x + e2y * e2y);
  float l_avg = (l1 + l2) / 2;

  return 2 / (l_avg*fmin(l1, l2) + fmax(l1, l2)/l_avg);
}

float FdebBenchmark::PositionCompatibility(int i, int j) { 
  float e1x1 = edge_src_x_[i];
  float e1y1 = edge_src_y_[i];
  float e1x2 = edge_dst_x_[i];
  float e1y2 = edge_dst_y_[i];
  float e2x1 = edge_src_x_[j];
  float e2y1 = edge_src_y_[j];
  float e2x2 = edge_dst_x_[j];
  float e2y2 = edge_dst_y_[j];
  float e1x = e1x2 - e1x1;
  float e1y = e1y2 - e1y1;
  float e2x = e2x2 - e2x1;
  float e2y = e2y2 - e2y1;
  float l1 = sqrt(e1x * e1x + e1y * e1y);
  float l2 = sqrt(e2x * e2x + e2y * e2y);
  float l_avg = (l1 + l2) / 2;

  float m1x = (e1x1 + e1x2) / 2;
  float m1y = (e1y1 + e1y2) / 2;
  float m2x = (e2x1 + e2x2) / 2;
  float m2y = (e2y1 + e2y2) / 2;
  float m_dist = sqrt((m2x - m1x)*(m2x - m1x) + (m2y - m1y)*(m2y - m1y));

  return l_avg / (l_avg + m_dist);
}

float FdebBenchmark::VisibilityCompatibility(int i, int j) { 
  return 1; 
}

void FdebBenchmark::BundlingCpu() {
  int num_point = 0;
  int iter = init_iter_count_;
  float step = init_step_size_;

  InitSubdivisionPoint();
  for (int i = 0; i < num_cycles_; i++) {
    printf("Cycle %d\n", i);
    num_point = GenerateSubdivisionPoint(num_point);
    InitForce(num_point);

    for (int j = 0; j < iter; j++) {
      printf("\tIter %d\n", j);
      BundlingIterCpu(num_point, step);
    }

    step = step / 2.0;
    iter = iter * 2 / 3;
  }
}

void FdebBenchmark::InitSubdivisionPoint() {
  point_x_.resize(edge_count_);
  point_y_.resize(edge_count_);
  for (int i = 0; i < edge_count_; i++) {
    point_x_[i].push_back(edge_src_x_[i]);
    point_y_[i].push_back(edge_src_y_[i]);
    point_x_[i].push_back(edge_dst_x_[i]);
    point_y_[i].push_back(edge_dst_y_[i]);
  }
}

int FdebBenchmark::GenerateSubdivisionPoint(int num_point) {
  for (int i = 0; i < edge_count_; i++) {
    auto iter_x = point_x_[i].begin();
    auto iter_y = point_y_[i].begin();

    for (int j = 0; j < num_point + 1; j++) {
      float x = (*iter_x + *(iter_x + 1)) / 2;
      float y = (*iter_y + *(iter_y + 1)) / 2;

      iter_x = point_x_[i].insert((iter_x + 1), x);
      iter_y = point_y_[i].insert((iter_y + 1), y);

      iter_x++;
      iter_y++;
    }
  }

  return num_point + num_point + 1;
}

void FdebBenchmark::InitForce(int num_point) {
  force_x_.resize(edge_count_);
  force_y_.resize(edge_count_);

  for (int i = 0; i < edge_count_; i++) {
    force_x_[i].resize(num_point + 2);
    force_y_[i].resize(num_point + 2);
  }
}

void FdebBenchmark::BundlingIterCpu(int num_point, float step) {
  UpdateForceCpu(num_point);
  MovePointsCpu(num_point, step);
}

void FdebBenchmark::UpdateForceCpu(int num_point) {
  for (int i = 0; i < edge_count_; i++) {
    for (int k = 1; k <= num_point; k++) {

      for (int j = 0; j < edge_count_; j++) {
        if (j == i) continue;
        float x1 = point_x_[i][k];
        float y1 = point_y_[i][k];
        float x2 = point_x_[j][k];
        float y2 = point_y_[j][k];
        float compatibility = compatibility_[i][j];
        float dist = sqrt( (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

        if (dist > 0) {
          float x = x2 - x1;
          float y = y2 - y1;
          // x = x / dist;
          // y = y / dist;

          force_x_[i][k] += x / dist / dist * compatibility;
          force_y_[i][k] += y / dist / dist * compatibility;
        }
      }

      // Self force
      float x = point_x_[i][k];
      float y = point_y_[i][k];
      float x_p = point_x_[i][k-1];
      float y_p = point_y_[i][k-1];
      float x_n = point_x_[i][k+1];
      float y_n = point_y_[i][k+1];

      force_x_[i][k] += kp_ * (x_p - x);
      force_y_[i][k] += kp_ * (y_p - y);
      force_x_[i][k] += kp_ * (x_n - x);
      force_y_[i][k] += kp_ * (y_n - y);

      // Normalize
      float mag = sqrt(force_x_[i][k] * force_x_[i][k] + 
                       force_y_[i][k] * force_y_[i][k]);
      force_x_[i][k] /= mag;
      force_y_[i][k] /= mag;
    }
  }
}

void FdebBenchmark::MovePointsCpu(int num_point, float step) {
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 1; j <= num_point; j++) {
      point_x_[i][j] += step * force_x_[i][j];
      point_y_[i][j] += step * force_y_[i][j];
    }
  }
}

void FdebBenchmark::PrintSubdevisedEdges() {
  for (int i = 0; i < edge_count_; i++) {
    printf("[%d] ", i);
    for (int j = 0; j < point_x_[i].size(); j++) {
      printf("(%.2f, %.2f)->", point_x_[i][j], point_y_[i][j]);
    }
    printf("\n");
  }
}

void FdebBenchmark::SaveSubdevisedEdges(const std::string &filename) {
  std::ofstream output;
  output.open(filename);
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 0; j < point_x_[i].size(); j++) {
      output << point_x_[i][j] << "," << point_y_[i][j] << ",";
    }
    output << "\n";
  }
  output.close();
}

void FdebBenchmark::Summarize() {}

void FdebBenchmark::Cleanup() {}
