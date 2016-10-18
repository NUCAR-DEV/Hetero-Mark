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
#include <cmath>
#include "src/ep/ep_benchmark.h"

void EpBenchmark::Initialize() {
  srand(kSeed);
  for (int i = 0; i < kNumVariables; i++) {
    fitness_function_[i] = 1.0 * rand() / RAND_MAX;
  }
}

void EpBenchmark::Verify() {
  srand(kSeed);
  islands_1_.clear();
  islands_2_.clear();
  for (int i = 0; i < max_generation_; i++) {
    Reproduce();
    Evaluate();
    Select();

    printf("Gen: %d, island 1 best: %f, island 2 best %f\n", i,
           islands_1_[0].fitness, islands_2_[0].fitness);

    cpu_result_island_1_ = islands_1_[0].fitness;
    cpu_result_island_2_ = islands_2_[0].fitness;

    Crossover();
    Mutate();
  }

  bool has_error = false;
  if (fabs(cpu_result_island_1_ - result_island_1_) >= 0.001) {
    has_error = true;
    printf("In island 1, expected to get %f, but get %f\n",
           cpu_result_island_1_, result_island_1_);
  }
  if (fabs(cpu_result_island_2_ - result_island_2_) >= 0.001) {
    has_error = true;
    printf("In island 2, expected to get %f, but get %f\n",
           cpu_result_island_2_, result_island_2_);
  }
  if (!has_error) {
    printf("Passed!\n");
  }
}

void EpBenchmark::Reproduce() {
  ReproduceInIsland(islands_1_);
  ReproduceInIsland(islands_2_);
}

void EpBenchmark::ReproduceInIsland(std::vector<Creature> &island) {
  while (island.size() < population_ / 2) {
    Creature creature = CreateRandomCreature();
    island.push_back(creature);
  }
}

EpBenchmark::Creature EpBenchmark::CreateRandomCreature() {
  Creature creature;
  for (int i = 0; i < kNumVariables; i++) {
    creature.parameters[i] = 1.0 * rand() / RAND_MAX;
  }
  return creature;
}

void EpBenchmark::Evaluate() {
  for (auto &creature : islands_1_) {
    ApplyFitnessFunction(creature);
  }

  for (auto &creature : islands_2_) {
    ApplyFitnessFunction(creature);
  }
}

void EpBenchmark::ApplyFitnessFunction(Creature &creature) {
  double fitness = 0;
  for (int i = 0; i < kNumVariables; i++) {
    fitness += pow(creature.parameters[i], i + 1) * fitness_function_[i];
  }
  creature.fitness = fitness;
}

void EpBenchmark::Select() {
  SelectInIsland(islands_1_);
  SelectInIsland(islands_2_);
}

void EpBenchmark::SelectInIsland(std::vector<Creature> &island) {
  auto comparator =
      [](Creature &a, Creature &b) { return b.fitness < a.fitness; };

  std::sort(island.begin(), island.end(), comparator);
  for (int i = 0; i < kNumEliminate / 2; i++) {
    island.pop_back();
  }
}

void EpBenchmark::Crossover() {
  CrossoverInIsland(islands_1_);
  CrossoverInIsland(islands_2_);
}

void EpBenchmark::CrossoverInIsland(std::vector<Creature> &island) {
  std::vector<Creature> new_creatures;
  for (auto &creature : island) {
    Creature best_creature = island[rand() % 10];
    Creature offspring;
    for (int i = 0; i < kNumVariables; i++) {
      if (rand() % 2 == 0) {
        offspring.parameters[i] = best_creature.parameters[i];
      } else {
        offspring.parameters[i] = creature.parameters[i];
      }
    }
    new_creatures.push_back(offspring);
  }
  island = new_creatures;
}

void EpBenchmark::Mutate() {
  for (int i = 0; i < islands_1_.size(); i++) {
    if (i % 7 != 0) continue;
    islands_1_[i].parameters[i % kNumVariables] *= 0.5;
  }

  for (int i = 0; i < islands_2_.size(); i++) {
    if (i % 7 != 0) continue;
    islands_2_[i].parameters[i % kNumVariables] *= 0.5;
  }
}

void EpBenchmark::Summarize() {}

void EpBenchmark::Cleanup() {}
