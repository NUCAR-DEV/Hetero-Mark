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

void FdebBenchmark::Verify() {
  FdebCpu();
  // InitSubdivisionPoint();
  // PrintSubdevisedEdges();
  // GenerateSubdivisionPoint();
  // PrintSubdevisedEdges();
}

void FdebBenchmark::FdebCpu() {
  CalculateCompatibility();
  BundlingCpu();
  SaveSubdevisedEdges("out_cpu.data");
}

void FdebBenchmark::CalculateCompatibility() {
  compatibility_.resize(edge_count_ * edge_count_);
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 0; j < edge_count_; j++) {
      if (i == j) {
        compatibility_[i * edge_count_ + j] = 1;
        continue;
      }

      // Fill in the real algorithm here.
      compatibility_[i * edge_count_ + j] =
          1.0 * AngleCompatibility(i, j) * ScaleCompatibility(i, j) *
          PositionCompatibility(i, j) * VisibilityCompatibility(i, j);
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

  return 2 / (l_avg * fmin(l1, l2) + fmax(l1, l2) / l_avg);
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
  float m_dist = sqrt((m2x - m1x) * (m2x - m1x) + (m2y - m1y) * (m2y - m1y));

  return l_avg / (l_avg + m_dist);
}

float FdebBenchmark::VisibilityCompatibility(int i, int j) { return 1; }

void FdebBenchmark::BundlingCpu() {
  int iter = init_iter_count_;
  step_size_ = init_step_size_;

  InitSubdivisionPoint();
  for (int i = 0; i < num_cycles_; i++) {
    printf("CPU Cycle %d\n", i);
    GenerateSubdivisionPoint();
    InitForce();

    for (int j = 0; j < iter; j++) {
      printf("\tCPU Iter %d\n", j);
      BundlingIterCpu();
    }

    step_size_ = step_size_ / 2.0;
    iter = iter * 2 / 3;
  }
}

void FdebBenchmark::InitSubdivisionPoint() {
  num_subpoint_ = 0;
  col_ = 2;
  // point_x_.resize(edge_count_ * 2);
  // point_y_.resize(edge_count_ * 2);
  point_x_.clear();
  point_y_.clear();
  for (int i = 0; i < edge_count_; i++) {
    point_x_.push_back(edge_src_x_[i]);
    point_y_.push_back(edge_src_y_[i]);
    point_x_.push_back(edge_dst_x_[i]);
    point_y_.push_back(edge_dst_y_[i]);
  }
}

void FdebBenchmark::GenerateSubdivisionPoint() {
  std::vector<float> old_x(point_x_.begin(), point_x_.end());
  std::vector<float> old_y(point_y_.begin(), point_y_.end());
  point_x_.resize(edge_count_ * (2 * num_subpoint_ + 3));
  point_y_.resize(edge_count_ * (2 * num_subpoint_ + 3));

  int old_index = 0;
  int index = 0;
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 0; j < num_subpoint_ + 1; j++) {
      point_x_[index] = old_x[old_index];
      point_y_[index] = old_y[old_index];
      index++;

      point_x_[index] = (old_x[old_index] + old_x[old_index + 1]) / 2;
      point_y_[index] = (old_y[old_index] + old_y[old_index + 1]) / 2;
      index++;
      old_index++;
    }

    // Do not forget the last point
    point_x_[index] = old_x[old_index];
    point_y_[index] = old_y[old_index];
    index++;
    old_index++;
  }

  num_subpoint_ += num_subpoint_ + 1;
  col_ = num_subpoint_ + 2;
}

void FdebBenchmark::InitForce() {
  force_x_.resize(edge_count_ * col_);
  force_y_.resize(edge_count_ * col_);
}

void FdebBenchmark::BundlingIterCpu() {
  UpdateForceCpu();
  MovePointsCpu();
}

void FdebBenchmark::UpdateForceCpu() {
  for (int i = 0; i < edge_count_; i++) {
    for (int k = 1; k <= num_subpoint_; k++) {
      float force_x = 0;
      float force_y = 0;

      for (int j = 0; j < edge_count_; j++) {
        if (j == i) continue;
        float x1 = point_x_[i * col_ + k];
        float y1 = point_y_[i * col_ + k];
        float x2 = point_x_[j * col_ + k];
        float y2 = point_y_[j * col_ + k];
        float compatibility = compatibility_[i * edge_count_ + j];
        float dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

        if (dist > 0) {
          float x = x2 - x1;
          float y = y2 - y1;

          force_x += x / dist / dist * compatibility;
          force_y += y / dist / dist * compatibility;
        }
      }

      // Self force
      float x = point_x_[i * col_ + k];
      float y = point_y_[i * col_ + k];
      float x_p = point_x_[i * col_ + k - 1];
      float y_p = point_y_[i * col_ + k - 1];
      float x_n = point_x_[i * col_ + k + 1];
      float y_n = point_y_[i * col_ + k + 1];

      force_x += kp_ * (x_p - x);
      force_y += kp_ * (y_p - y);
      force_x += kp_ * (x_n - x);
      force_y += kp_ * (y_n - y);

      // Normalize
      float mag = sqrt(force_x * force_x + force_y * force_y);
      force_x /= mag;
      force_y /= mag;

      force_x_[i * col_ + k] = force_x;
      force_y_[i * col_ + k] = force_y;
    }
  }
}

void FdebBenchmark::MovePointsCpu() {
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 1; j <= num_subpoint_; j++) {
      point_x_[i * col_ + j] += step_size_ * force_x_[i * col_ + j];
      point_y_[i * col_ + j] += step_size_ * force_y_[i * col_ + j];
    }
  }
}

void FdebBenchmark::PrintSubdevisedEdges() {
  for (int i = 0; i < edge_count_; i++) {
    printf("[%d] ", i);
    for (int j = 0; j < num_subpoint_ + 1; j++) {
      printf("(%.2f, %.2f)->", point_x_[i * col_ + j], point_y_[i * col_ + j]);
    }
    printf("(%.2f, %.2f)", point_x_[i * col_ + col_ - 1],
           point_y_[i * col_ + +col_ - 1]);
    printf("\n");
  }
}

void FdebBenchmark::SaveSubdevisedEdges(const std::string &filename) {
  std::ofstream output;
  output.open(filename);
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 0; j < num_subpoint_ + 1; j++) {
      output << point_x_[i * col_ + j] << "," << point_y_[i * col_ + j] << ",";
    }
    output << point_x_[i * col_ + col_ - 1] << ","
           << point_y_[i * col_ + col_ - 1] << "\n";
  }
  output.close();
}

void FdebBenchmark::Summarize() {}

void FdebBenchmark::Cleanup() {}
