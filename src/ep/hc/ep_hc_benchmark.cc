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

#include "src/ep/hc/ep_hc_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

void EpHcBenchmark::Initialize() { EpBenchmark::Initialize(); }

void EpHcBenchmark::Run() {
  if (pipelined_) {
    PipelinedRun();
  } else {
    NormalRun();
  }
}

void EpHcBenchmark::PipelinedRun() {
  seed_ = kSeedInitValue;
  ReproduceInIsland(islands_1_);
  for (int i = 0; i < max_generation_; i++) {
    timer_->Start();
    std::thread t1(&EpHcBenchmark::ReproduceInIsland, this,
                   std::ref(islands_2_));
    std::thread t2(&EpHcBenchmark::EvaluateGpu, this, std::ref(islands_1_));
    t1.join();
    t2.join();
    timer_->End({"Stage 1"});

    timer_->Start();
    std::thread t3(&EpHcBenchmark::EvaluateGpu, this, std::ref(islands_2_));
    std::thread t4(&EpHcBenchmark::SelectInIsland, this, std::ref(islands_1_));
    t4.join();
    result_island_1_ = islands_1_[0].fitness;
    std::thread t5(&EpHcBenchmark::CrossoverInIsland, this,
                   std::ref(islands_1_));
    t5.join();
    t3.join();
    timer_->End({"Stage 2"});

    timer_->Start();
    std::thread t6(&EpHcBenchmark::SelectInIsland, this, std::ref(islands_2_));
    std::thread t7(&EpHcBenchmark::MutateGpu, this, std::ref(islands_1_));
    t6.join();
    result_island_2_ = islands_2_[0].fitness;
    std::thread t8(&EpHcBenchmark::CrossoverInIsland, this,
                   std::ref(islands_2_));
    t7.join();
    t8.join();
    timer_->End({"Stage 3"});

    timer_->Start();
    std::thread t9(&EpHcBenchmark::MutateGpu, this, std::ref(islands_2_));
    std::thread t10(&EpHcBenchmark::ReproduceInIsland, this,
                    std::ref(islands_1_));
    t9.join();
    t10.join();
    timer_->End({"Stage 4"});

    timer_->Summarize();
  }
}

void EpHcBenchmark::NormalRun() {
  seed_ = kSeedInitValue;
  for (int i = 0; i < max_generation_; i++) {
    Reproduce();
    EvaluateGpu(islands_1_);
    EvaluateGpu(islands_2_);
    Select();

    result_island_1_ = islands_1_[0].fitness;
    result_island_2_ = islands_2_[0].fitness;

    Crossover();
    MutateGpu(islands_1_);
    MutateGpu(islands_2_);
  }
}

void EpHcBenchmark::EvaluateGpu(std::vector<Creature> &island) {
  hc::array_view<Creature, 1> av_island(island.size(), island);
  hc::array_view<double, 1> av_fitness_func(kNumVariables, fitness_function_);
  hc::parallel_for_each(hc::extent<1>(island.size()),
                        [=](hc::index<1> i)[[hc]] {
                          double fitness = 0;
                          Creature &creature = av_island[i];
                          for (int j = 0; j < kNumVariables; j++) {
                            double pow = 1;
                            for (int k = 0; k < j + 1; k++) {
                              pow *= creature.parameters[j];
                            }
                            fitness += pow * av_fitness_func[j];
                          }
                          creature.fitness = fitness;
                        });
  av_island.synchronize();
}

void EpHcBenchmark::MutateGpu(std::vector<Creature> &island) {
  hc::array_view<Creature, 1> av_island(island.size(), island);
  hc::parallel_for_each(hc::extent<1>(island.size()),
                        [=](hc::index<1> i)[[hc]] {
                          if (i[0] % 7 != 0) return;
                          av_island[i].parameters[i[0] % kNumVariables] *= 0.5;
                        });
  av_island.synchronize();
}

void EpHcBenchmark::Cleanup() { EpBenchmark::Cleanup(); }
