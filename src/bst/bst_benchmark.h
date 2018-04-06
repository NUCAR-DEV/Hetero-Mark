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

#ifndef SRC_BST_BST_BENCHMARK_H_
#define SRC_BST_BST_BENCHMARK_H_

#include <atomic>
#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

#define UM_MUTEX_LOCK 1
#define UM_MUTEX_UNLOCK 0

typedef struct { std::atomic<int> count; } um_mutex;

typedef struct BinTree {
  int64_t value;           // Value at a node
  struct BinTree *left;    // Pointer to the left node
  struct BinTree *right;   // Pointer to the right node
  struct BinTree *parent;  // Pointer to the parent node
  um_mutex mutex_node;
  int childDevType;  // Indicates which device inserted its child nodes
  int visited;       // Indicates whether the node is inserted to binary tree
} Node;

class BstBenchmark : public Benchmark {
 protected:
  Node *tree_buffer_;
  Node *root_;
  uint32_t seed_;
  uint32_t total_nodes_;
  uint32_t host_percentage_ = 30;
  uint32_t host_nodes_;
  uint32_t device_nodes_;
  uint32_t num_insert_ = 200;
  uint32_t init_tree_insert_ = 10;
  uint32_t CountNodes(Node *root);
  void InitializeNodes(Node *data, uint32_t num_nodes, uint32_t seed);
  void InsertNode(Node *nextData, Node *root);
  Node *MakeBinaryTree(uint32_t num_nodes, Node *inroot);
  void UmMutexInit(um_mutex *lock, int value);
  void UmMutexLock(um_mutex *lock);
  void UmMutexUnlock(um_mutex *lock);

 public:
  BstBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void SetNumNodes(uint32_t num_insert) { num_insert_ = num_insert; }
  void SetInitPosition(uint32_t init_tree_insert) {
    init_tree_insert_ = init_tree_insert;
  }
  void SetHostPercentage(uint32_t host_percentage) {
    host_percentage_ = host_percentage;
  }
  // Getters
  uint32_t GetTotalNodes() { return total_nodes_; }
  Node *GetTreeBuff() { return tree_buffer_; }
};

#endif  // SRC_BST_BST_BENCHMARK_H_
