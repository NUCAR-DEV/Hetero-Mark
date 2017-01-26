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
#include <climits>
#include <iostream>
#include <fstream>
#include <string>
#include "src/ga/ga_benchmark.h"

void GaBenchmark::Initialize() {
  std::ifstream input(input_file_);
  if (input.is_open()) {
    getline(input, query_sequence_);
    getline(input, target_sequence_);
    input.close();
  } else {
    std::cerr << "Failed to open input file at " << input_file_ << "\n";
    exit(0);
  }

  if (coarse_match_length_ > query_sequence_.length()) {
    coarse_match_length_ = query_sequence_.length();
    coarse_match_threshold_ = 2;
  }

  // printf("Coarse match length: %d\n", coarse_match_length_);
  // printf("Coarse match threshold: %d\n", coarse_match_threshold_);
}

void GaBenchmark::Verify() {
  CoarseMatch();

  for (int pos : coarse_match_position_) {
    int start = pos - query_sequence_.length();
    if (start < 0) start = 0;

    int end = pos + query_sequence_.length();
    if (end > target_sequence_.length()) end = target_sequence_.length();

    FineMatch(start, end, cpu_matches_);
  }
}

void GaBenchmark::CoarseMatch() {
  int max_searchable_length = target_sequence_.length() - coarse_match_length_;
  for (uint32_t i = 0; i < max_searchable_length; i++) {
    if (CoarseMatchAtTargetPosition(i)) {
      coarse_match_position_.push_back(i);
    }
  }
}

bool GaBenchmark::CoarseMatchAtTargetPosition(int target_index) {
  int max_searchable_length = query_sequence_.length() - coarse_match_length_;
  for (uint32_t i = 0; i <= max_searchable_length; i++) {
    int distance =
        HammingDistance(target_sequence_.c_str() + target_index,
                        query_sequence_.c_str() + i, coarse_match_length_);

    if (distance < coarse_match_threshold_) return true;
  }
  return false;
}

int GaBenchmark::HammingDistance(const char *seq1, const char *seq2,
                                 int length) {
  int distance = 0;
  for (int i = 0; i < length; i++) {
    if (seq1[i] != seq2[i]) distance++;
  }

  /*
  printf("seq1: ");
  for (int i = 0; i < length; i++) {
    printf("%c", seq1[i]);
  }
  printf("\n");

  printf("seq2: ");
  for (int i = 0; i < length; i++) {
    printf("%c", seq2[i]);
  }
  printf("\n");

  printf("Distance: %d\n", distance);
  */

  return distance;
}

void GaBenchmark::FineMatch(int start, int end, std::list<Match *> &matches) {
  int target_length = end - start;
  int query_length = query_sequence_.length();

  Matrix score_matrix;
  Matrix action_matrix;
  CreateMatrix(&score_matrix, query_length, target_length);
  CreateMatrix(&action_matrix, query_length, target_length);

  for (int i = 0; i < query_length; i++) {
    for (int j = 0; j < target_length; j++) {
      FillCell(score_matrix, action_matrix, i, j, start);
    }
  }

  /*
  printf("   ");
  for (int i = 0; i < query_length; i++) {
    printf(" %c ", query_sequence_[i]);
  }
  printf("\n");
  for (int i = 0; i < target_length; i++) {
    printf(" %c ", target_sequence_[i + start]);
    for (int j = 0; j < query_length; j++) {
      printf("%3d", score_matrix[j][i]);
    }
    printf("\n");
  }
  */

  Match *match = GenerateMatch(score_matrix, action_matrix, start, end);
  matches.push_back(match);

  DestroyMatrix(&score_matrix, query_length, target_length);
  DestroyMatrix(&action_matrix, query_length, target_length);
}

void GaBenchmark::CreateMatrix(Matrix *matrix, int x, int y) {
  *matrix = new int *[x];
  for (int i = 0; i < x; i++) {
    (*matrix)[i] = new int[y]();
    for (int j = 0; j < y; j++) {
      (*matrix)[i][j] = 0;
    }
  }
}

void GaBenchmark::DestroyMatrix(Matrix *matrix, int x, int y) {
  for (int i = 0; i < x; i++) {
    delete[](*matrix)[i];
  }
  delete[](*matrix);
}

void GaBenchmark::FillCell(Matrix score_matrix, Matrix action_matrix, int x,
                           int y, int target_offset) {
  int local_score = 0;
  if (query_sequence_[x] != target_sequence_[target_offset + y]) {
    local_score = -mismatch_penalty;
  } else {
    local_score = match_reward;
  }

  if (x == 0 && y == 0) {
    score_matrix[x][y] = local_score;
    action_matrix[x][y] = 2;
  } else if (x == 0) {
    action_matrix[x][y] = 1;
    score_matrix[x][y] = local_score;
  } else if (y == 0) {
    action_matrix[x][y] = 3;
    score_matrix[x][y] = local_score;
  } else {
    int up_score = score_matrix[x][y - 1] - gap_penalty;
    int diag_score = local_score + score_matrix[x - 1][y - 1];
    int left_score = score_matrix[x - 1][y] - gap_penalty;

    if (up_score > diag_score && up_score > left_score) {
      score_matrix[x][y] = up_score;
      action_matrix[x][y] = 1;
    } else if (diag_score >= up_score && diag_score >= left_score) {
      score_matrix[x][y] = diag_score;
      action_matrix[x][y] = 2;
    } else {
      score_matrix[x][y] = left_score;
      action_matrix[x][y] = 3;
    }
  }
}

GaBenchmark::Match *GaBenchmark::GenerateMatch(Matrix score_matrix,
                                               Matrix action_matrix,
                                               int target_start,
                                               int target_end) {
  int target_length = target_end - target_start;
  int query_length = query_sequence_.length();

  int exit_x = query_length - 1;
  int exit_y = 0;
  int highest_score = INT_MIN;
  for (int i = 0; i < target_length; i++) {
    if (score_matrix[exit_x][i] > highest_score) {
      highest_score = score_matrix[exit_x][i];
      exit_y = i;
    }
  }

  // std::cout << "Exit point: " << exit_x << ", " << exit_y << "\n";

  Match *match = new Match();
  int curr_x = exit_x;
  int curr_y = exit_y;
  while (curr_x > 0) {
    int action = action_matrix[curr_x][curr_y];
    match->directions.push_front(action);
    if (action == 1) {
      curr_y -= 1;
    } else if (action == 2) {
      curr_x -= 1;
      curr_y -= 1;
    } else if (action == 3) {
      curr_x -= 1;
    }
  }

  match->target_index = target_start + curr_y;
  match->similarity = highest_score;
  /*
  std::cout << "Start index: " << target_start << "\n";
  std::cout << "Target index: " << match->target_index << "\n";
  std::cout << "Actions: ";
  for (int action : match->directions) {
    std::cout << action;
  }
  std::cout << "\n";
  */

  return match;
}

void GaBenchmark::Summarize() {
  std::cout << "Target Sequence:";
  for (uint32_t i = 0; i < target_sequence_.length(); i++) {
    if (i % 64 == 0) std::cout << "\n\t";
    std::cout << target_sequence_[i];
  }
  std::cout << "\n";

  std::cout << "Query Sequence:";
  for (uint32_t i = 0; i < query_sequence_.length(); i++) {
    if (i % 64 == 0) std::cout << "\n\t";
    std::cout << query_sequence_[i];
  }
  std::cout << "\n";

  std::cout << "Coarse Match:";
  for (uint32_t i = 0; i < coarse_match_position_.size(); i++) {
    if (i % 12 == 0) std::cout << "\n\t";
    std::cout << coarse_match_position_[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "Final Match:\n";
  for (auto match : matches_) {
    std::cout << match->target_index << "(" << match->similarity << ")\n";

    int target_index = match->target_index;
    int query_index = 0;

    int action = match->directions.front();
    if (action == 3) {
      printf("|   %c", query_sequence_[query_index]);
      query_index++;
    } else if (action == 2) {
      if (target_sequence_[target_index] == query_sequence_[query_index]) {
        printf("%c - %c", target_sequence_[target_index],
               query_sequence_[query_index]);
      } else {
        printf("%c   %c", target_sequence_[target_index],
               query_sequence_[query_index]);
      }

      target_index++;
      query_index++;
    }
    std::cout << "\n";

    for (int action : match->directions) {
      if (action == 1) {
        printf("%c   | ", target_sequence_[target_index]);
        target_index++;
      } else if (action == 2) {
        if (target_sequence_[target_index] == query_sequence_[query_index]) {
          printf("%c - %c", target_sequence_[target_index],
                 query_sequence_[query_index]);
        } else {
          printf("%c   %c", target_sequence_[target_index],
                 query_sequence_[query_index]);
        }

        target_index++;
        query_index++;
      } else {
        printf("|   %c", query_sequence_[query_index]);
        query_index++;
      }

      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

void GaBenchmark::Cleanup() {}
