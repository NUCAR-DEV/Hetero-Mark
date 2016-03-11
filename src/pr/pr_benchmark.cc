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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cmath>
#include "src/pr/pr_benchmark.h"

void PrBenchmark::Initialize() {
  LoadInputFile();

  page_rank_ = new float[num_nodes_];
}

void PrBenchmark::LoadInputFile() {
  std::ifstream matrix_file;
  matrix_file.open(input_file_name_);

  // Check if the file is opened
  if (!matrix_file.good()) {
    std::cerr << "Cannot open input file\n";
    exit(-1);
  }

  // Load size information
  matrix_file >> num_connections_ >> num_nodes_;

  // Load row offsets
  row_offsets_ = new uint32_t[num_nodes_ + 1];
  for (uint32_t i = 0; i < num_nodes_ + 1; i++) {
    matrix_file >> row_offsets_[i];
  }

  // Load column numbers
  column_numbers_ = new uint32_t[num_connections_];
  for (uint32_t i = 0; i < num_connections_; i++) {
    matrix_file >> column_numbers_[i];
  }

  // Load values
  values_ = new float[num_connections_];
  for (uint32_t i = 0; i < num_connections_; i++) {
    matrix_file >> values_[i];
  }
}

void PrBenchmark::Verify() {
  uint32_t i;
  float *cpu_page_rank = new float[num_nodes_];
  float *cpu_page_rank_old = new float[num_nodes_];

  // Initialize CPU data
  for (i = 0; i < num_nodes_; i++) {
    cpu_page_rank[i] = 1.0 / num_nodes_;
    cpu_page_rank_old[i] = 0.0;
  }

  // Calculate On CPU
  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      CpuPageRankUpdate(cpu_page_rank, cpu_page_rank_old);
    } else {
      CpuPageRankUpdate(cpu_page_rank_old, cpu_page_rank);
    }
  }
  if (i % 2 == 0) {
    memcpy(cpu_page_rank, cpu_page_rank_old, num_nodes_ * sizeof(float));
  }

  // Compare with GPU result
  bool has_error = false;
  for (i = 0; i < num_nodes_; i++) {
    if (fabs(page_rank_[i] - cpu_page_rank[i]) > 1e-20) {
      printf("Error with node %i, expected to be %e, but was %e\n", i,
             cpu_page_rank[i], page_rank_[i]);
      has_error = true;
    }
  }
  if (!has_error) {
    printf("Passed.\n");
  }

  // Cleanup
  delete cpu_page_rank;
  delete cpu_page_rank_old;
}

void PrBenchmark::CpuPageRankUpdate(float *input, float *output) {
  for (uint32_t i = 0; i < num_nodes_; i++) {
    float new_value = 0;
    for (uint32_t j = row_offsets_[i]; j < row_offsets_[i + 1]; j++) {
      new_value += values_[j] * input[column_numbers_[j]];
    }
    output[i] = new_value;
  }
}

void PrBenchmark::Summarize() {
  printf("%d nodes, %d connections processed\n", num_nodes_, num_connections_);
  DumpConnections();
  DumpPageRanks(page_rank_);
}

void PrBenchmark::DumpConnections() {
  for (uint32_t i = 0; i < num_nodes_; i++) {
    printf("Node %d is referenced by: \n", i);
    for (uint32_t j = row_offsets_[i]; j < row_offsets_[i + 1]; j++) {
      printf("\tnode %d with strength %e, \n", column_numbers_[j], values_[j]);
    }
  }
}

void PrBenchmark::DumpPageRanks(float *page_rank) {
  for (uint32_t i = 0; i < num_nodes_; i++) {
    printf("Node %d gets value %e\n", i, page_rank[i]);
  }
}

void PrBenchmark::Cleanup() {
  delete row_offsets_;
  delete column_numbers_;
  delete values_;
  delete page_rank_;
}
