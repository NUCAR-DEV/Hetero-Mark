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
#include <sstream>
#include <vector>
#include "src/kmeans/kmeans_benchmark.h"

void KmeansBenchmark::Initialize() {
  std::ifstream file(filename_);
  std::string line;
  std::vector<std::vector<unsigned>> points;

  // Read points from input file
  while (std::getline(file, line)) {
    num_points_++;

    std::vector<unsigned> features;
    std::stringstream ss(line);
    unsigned n;

    // Ignore the 1st attribute
    bool ignore = true;
    ss >> n;
    while (ss.good()) {
      if (!ignore) features.push_back(n);
      ignore = false;
      ss >> n;
    }
    points.push_back(features);
  }

  // Count number of points and features
  num_points_ = points.size();
  num_features_ = points[0].size();

  // Sanity check
  if (num_points_ < min_num_clusters_)
    std::cerr << "Error: More clusters than points" << std::endl;

  // Copy data to host buffer
  // host_features_ = Misc::NewUniqueArray<float>(num_points_ * num_features_);
  // std::memcpy(host_features_.get(), points.data(),
  //             num_points_ * num_features_ * sizeof(float));
}

void KmeansBenchmark::Verify() {}

void KmeansBenchmark::Summarize() {}

void KmeansBenchmark::Cleanup() {}
