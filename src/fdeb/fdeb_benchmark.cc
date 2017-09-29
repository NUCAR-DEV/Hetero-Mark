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

#include <string>
#include <fstream>

#include "src/fdeb/fdeb_benchmark.h"

void FdebBenchmark::Initialize() {
  LoadNodeData(data_name_ + "_node.data");
  LoadEdgeData(data_name_ + "_edge.data");
}

void FdebBenchmark::LoadNodeData(const std::string &file_name) {
  std::string value_str;
  float value;
  std::ifstream file(file_name);

  while(file.good()) {
    getline(file, value_str, ',');
    if (value_str == "") break;
    std::cout << value_str << ",";
    value = std::stof(value_str); 
    node_x_.push_back(value);

    getline(file, value_str);
    if (value_str == "") break;
    std::cout << value_str << "\n";
    value = std::stof(value_str); 
    node_y_.push_back(value);

    node_count_++;
  }
}

void FdebBenchmark::LoadEdgeData(const std::string &file_name) {
  std::string value_str;
  int value;
  std::ifstream file(file_name);

  while(file.good()) {
    getline(file, value_str, ',');
    if (value_str == "") break;
    std::cout << value_str << ",";
    value = std::stoi(value_str); 
    edge_src_.push_back(value);

    getline(file, value_str);
    if (value_str == "") break;
    std::cout << value_str << "\n";
    value = std::stof(value_str); 
    edge_dst_.push_back(value);

    edge_count_++;
  }

  PrintEdges();
}

void FdebBenchmark::PrintNodes() {
  for (int i = 0; i < node_count_; i++) {
    printf("[%d] (%f, %f)\n", i, node_x_[i], node_y_[i]);
  }
}

void FdebBenchmark::PrintEdges() {
  for (int i = 0; i < edge_count_; i++) {
    int src = edge_src_[i];
    int dst = edge_dst_[i];
    printf("[%d] %d(%f, %f) -> %d(%f, %f)\n", 
        i, 
        src, node_x_[src], node_y_[src],
        dst, node_x_[dst], node_y_[dst]);
  }
}

void FdebBenchmark::Verify() {
  FdebCpu();
  // int num_point = 0;
  // InitSubdivisionPoint();
  // for(int i = 0; i < 7; i++) {
  //   num_point = GenerateSubdivisionPoint(num_point);
  //   PrintSubdevisedEdges();
  // }
}

void FdebBenchmark::FdebCpu() {
  CalculateCompatibility(); 
  BundlingCpu();
}

void FdebBenchmark::CalculateCompatibility() {
  compatibility_.resize(edge_count_ * edge_count_);
  for (int i = 0; i < edge_count_; i++) {
    for (int j = 0; j < edge_count_; j++) {
      int index = i * edge_count_ + j;
      if (i == j) {
        compatibility_[index] = 1;
        continue;
      }

      // Fill in the real algorithm here.
      compatibility_[index] = 1;
    }
  }
}

void FdebBenchmark::BundlingCpu() {
  int num_point = 0;
  int iter = init_iter_count_;
  float step = init_step_size_;
  
  InitSubdivisionPoint();
  for (int i = 0; i < num_cycles_; i++) {
    num_point = GenerateSubdivisionPoint(num_point);

    for (int j = 0; j < iter; j++) {
    }

    step = step/2.0;
    iter = iter*2/3;
  }
}

void FdebBenchmark::InitSubdivisionPoint() {
  point_x_.resize(edge_count_);
  point_y_.resize(edge_count_);
  for (int i = 0; i < edge_count_; i++) {
    int src = edge_src_[i];
    int dst = edge_dst_[i];

    point_x_[i].push_back(node_x_[src]);
    point_y_[i].push_back(node_y_[src]);
    point_x_[i].push_back(node_x_[dst]);
    point_y_[i].push_back(node_y_[dst]);
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

void FdebBenchmark::PrintSubdevisedEdges() {
  for (int i = 0; i < edge_count_; i++) {
    printf("[%d] ", i);
    for (int j = 0; j < point_x_[i].size(); j++) {
      printf("(%.2f, %.2f)->", point_x_[i][j], point_y_[i][j]);
    } 
    printf("\n");
  }
}

void FdebBenchmark::Summarize() {
}

void FdebBenchmark::Cleanup() {
}
