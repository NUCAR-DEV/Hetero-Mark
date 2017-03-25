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

#ifndef SRC_BE_BE_BENCHMARK_H_
#define SRC_BE_BE_BENCHMARK_H_

#include <vector>
// #include <libavcodec/avcodec.h>
#include <opencv2/opencv.hpp>

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class BeBenchmark : public Benchmark {
 protected:
  uint32_t num_frames_;
  uint32_t width_;
  uint32_t height_;
  bool collaborative_execution_;
  std::string input_file_;

  cv::VideoCapture video_;

  std::vector<uint8_t> data_;
  std::vector<uint8_t> foreground_;
  std::vector<float> background_;

  float alpha_ = 0.03;

  uint8_t *nextFrame();

 public:
  void Initialize() override;
  void Run() override {};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void SetInputFile(const std::string &input_file) { 
    input_file_ = input_file; 
  }
  void SetNumFrames(uint32_t num_frames) { num_frames_ = num_frames; }
  void SetCollaborativeExecution(bool collaborative_execution) {
    collaborative_execution_ = collaborative_execution;
  }
};

#endif  // SRC_BE_BE_BENCHMARK_H_
