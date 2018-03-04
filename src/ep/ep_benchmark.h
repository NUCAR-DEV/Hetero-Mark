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

#ifndef SRC_EP_EP_BENCHMARK_H_
#define SRC_EP_EP_BENCHMARK_H_

#include <vector>
#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

#define NUM_VARIABLES 500

class Creature {
 public:
  double fitness;
  double parameters[NUM_VARIABLES];

  void Dump();
};

class EpBenchmark : public Benchmark {
 protected:
  static const int32_t kNumVariables = NUM_VARIABLES;
  static const int32_t kNumEliminate = 0;
  static const int kSeedInitValue = 1;

  unsigned int seed_ = 1;

  uint32_t max_generation_;
  uint32_t population_;
  bool pipelined_;

  double fitness_function_[kNumVariables];
  std::vector<Creature> islands_1_;
  std::vector<Creature> islands_2_;
  double result_island_1_;
  double cpu_result_island_1_;
  double result_island_2_;
  double cpu_result_island_2_;

  void Reproduce();
  void ReproduceInIsland(std::vector<Creature> *island);
  Creature CreateRandomCreature();
  void Evaluate();
  void ApplyFitnessFunction(Creature *creature);
  void Select();
  void SelectInIsland(std::vector<Creature> *island);
  void Crossover();
  void CrossoverInIsland(std::vector<Creature> *island);
  void Mutate();

 public:
  EpBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void SetMaxGeneration(uint32_t max_generation) {
    max_generation_ = max_generation;
  }

  void SetPopulation(uint32_t population) { population_ = population; }

  void SetPipelined(bool pipelined) { pipelined_ = pipelined; }
};

#endif  // SRC_EP_EP_BENCHMARK_H_
