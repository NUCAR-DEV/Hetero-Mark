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
 * * Permission is hereby granted, free of charge, to any person obtaining a
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

#include "src/ga/hc/ga_hc_benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <hcc/hc.hpp>

void GaHcBenchmark::Initialize() {
  GaBenchmark::Initialize();
  coarse_match_result_ = new char[target_sequence_.length()]();
}

void GaHcBenchmark::Run() {
  if (collaborative_) {
    CollaborativeRun();
  } else {
    NonCollaborativeRun();
  }
}

void GaHcBenchmark::CollaborativeRun() {
  printf("Collaborative\n");
  std::vector<std::thread> threads;
  int max_searchable_length = target_sequence_.length() - coarse_match_length_;

  hc::array_view<char, 1> av_target(target_sequence_.length(),
                                    (char *)target_sequence_.c_str());
  hc::array_view<char, 1> av_query(query_sequence_.length(),
                                   (char *)query_sequence_.c_str());

  int current_position = 0;
  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};
    hc::array_view<char, 1> av_batch_result(kBatchSize, batch_result);

    int end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    int length = end_position - current_position;
    int coarse_match_length = coarse_match_length_;
    int coarse_match_threshold = coarse_match_threshold_;
    int query_sequence_length = query_sequence_.length();

    // av_batch_result.discard_data();
    hc::parallel_for_each(hc::extent<1>(length), [=](hc::index<1> index)[[hc]] {
      bool match = false;
      int max_length = query_sequence_length - coarse_match_length;
      for (uint32_t i = 0; i <= max_length; i++) {
        int distance = 0;
        for (int j = 0; j < coarse_match_length; j++) {
          if (av_target[current_position + index + j] != av_query[i + j]) {
            distance++;
          }
        }

        if (distance < coarse_match_threshold) {
          match = true;
          break;
        }
      }

      if (match) {
        av_batch_result[index] = 1;
      }
    });

    // av_batch_result.synchronize();
    // memcpy(coarse_match_result_ + current_position, batch_result, length);

    for (int i = 0; i < length; i++) {
      if (av_batch_result[i] == 1) {
        int start = i + current_position;
        int end = start + query_sequence_.length();
        if (end > target_sequence_.length()) {
          end = target_sequence_.length();
        }
        threads.push_back(std::thread(&GaHcBenchmark::FineMatch, this, start,
                                      end, std::ref(matches_)));
      }
    }

    current_position = end_position;
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void GaHcBenchmark::NonCollaborativeRun() {
  int max_searchable_length = target_sequence_.length() - coarse_match_length_;

  hc::array_view<char, 1> av_target(target_sequence_.length(),
                                    (char *)target_sequence_.c_str());
  hc::array_view<char, 1> av_query(query_sequence_.length(),
                                   (char *)query_sequence_.c_str());

  int current_position = 0;
  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};
    hc::array_view<char, 1> av_batch_result(kBatchSize, batch_result);

    int end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    int length = end_position - current_position;
    int coarse_match_length = coarse_match_length_;
    int coarse_match_threshold = coarse_match_threshold_;
    int query_sequence_length = query_sequence_.length();

    hc::parallel_for_each(hc::extent<1>(length), [=](hc::index<1> index)[[hc]] {
      bool match = false;
      int max_length = query_sequence_length - coarse_match_length;
      for (uint32_t i = 0; i <= max_length; i++) {
        int distance = 0;
        for (int j = 0; j < coarse_match_length; j++) {
          if (av_target[current_position + index + j] != av_query[i + j]) {
            distance++;
          }
        }

        if (distance < coarse_match_threshold) {
          match = true;
          break;
        }
      }

      if (match) {
        av_batch_result[index] = 1;
      }
    });

    av_batch_result.synchronize();
    memcpy(coarse_match_result_ + current_position, batch_result, length);

    current_position = end_position;
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < target_sequence_.length(); i++) {
    if (coarse_match_result_[i] == 1) {
      int end = i + query_sequence_.length();
      if (end > target_sequence_.length()) end = target_sequence_.length();
      threads.push_back(std::thread(&GaHcBenchmark::FineMatch, this, i, end,
                                    std::ref(matches_)));
    }
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void GaHcBenchmark::Cleanup() { GaBenchmark::Cleanup(); }
