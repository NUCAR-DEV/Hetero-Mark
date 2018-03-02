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

#ifndef SRC_GA_GA_BENCHMARK_H_
#define SRC_GA_GA_BENCHMARK_H_

#include <list>
#include <mutex>
#include <string>
#include <vector>
#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class GaBenchmark : public Benchmark {
 protected:
  class Match {
   public:
    int similarity;
    int target_index;
    std::list<int> directions;
  };

  std::string input_file_;
  bool collaborative_;

  std::vector<char> target_sequence_;
  std::vector<char> query_sequence_;

  uint32_t coarse_match_length_ = 11;
  uint32_t coarse_match_threshold_ = 1;

  int mismatch_penalty = 1;
  int gap_penalty = 2;
  int match_reward = 4;

  std::vector<int> coarse_match_position_;
  std::mutex match_mutex_;
  std::list<Match *> matches_;
  std::list<Match *> cpu_matches_;

  void CoarseMatch();
  bool CoarseMatchAtTargetPosition(int target_index);
  uint32_t HammingDistance(const char *seq1, const char *seq2, int length);

  typedef int **Matrix;
  void FineMatch(int start, int end, std::list<Match *> *matches);
  void FillCell(Matrix score_matrix, Matrix action_matrix, int i, int j,
                int target_offset);
  Match *GenerateMatch(Matrix score_matrix, Matrix action_matrix,
                       int target_start, int target_end);
  void CreateMatrix(Matrix *matrix, int x, int y);
  void DestroyMatrix(Matrix *matrix, int x, int y);

 public:
  GaBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void SetInputFile(const std::string &input_file) { input_file_ = input_file; }
  void SetCollaborativeExecution(bool collaborative) {
    collaborative_ = collaborative;
  }
};

#endif  // SRC_GA_GA_BENCHMARK_H_
