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

#include "src/bst/bst_benchmark.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void BstBenchmark::UmMutexInit(um_mutex *lock, int value) {
  atomic_store_explicit(&lock->count, value, std::memory_order_release);
}

void BstBenchmark::UmMutexLock(um_mutex *lock) {
  int expected = UM_MUTEX_UNLOCK;
  while (!atomic_compare_exchange_strong_explicit(
      &lock->count, &expected, UM_MUTEX_LOCK, std::memory_order_seq_cst,
      std::memory_order_seq_cst)) {
    expected = UM_MUTEX_UNLOCK;
  }
}

void BstBenchmark::UmMutexUnlock(um_mutex *lock) {
  atomic_store_explicit(&lock->count, UM_MUTEX_UNLOCK,
                        std::memory_order_release);
}

void BstBenchmark::InsertNode(Node *tmpData, Node *root) {
  Node *nextNode = root;
  Node *tmp_parent = NULL;
  Node *nextData;

  nextData = tmpData;
  int64_t key = nextData->value;
  int64_t flag = 0;
  int done = 0;

  while (nextNode) {
    tmp_parent = nextNode;
    flag = (key - (nextNode->value));
    nextNode = (flag < 0) ? nextNode->left : nextNode->right;
  }

  Node *child = nextNode;

  do {
    um_mutex *parent_mutex = &tmp_parent->mutex_node;
    UmMutexLock(parent_mutex);

    child = (flag < 0) ? tmp_parent->left : tmp_parent->right;

    if (child) {
      // Parent node has been updated since last check. Get the new parent and
      // iterate again
      tmp_parent = child;
    } else {
      // Insert the node
      tmp_parent->left = (flag < 0) ? nextData : tmp_parent->left;

      tmp_parent->right = (flag >= 0) ? nextData : tmp_parent->right;

      // Whether host only insert (childDevType=100) or both host and device
      // (childDevType=300)
      if (tmp_parent->childDevType == -1 || tmp_parent->childDevType == 100)
        tmp_parent->childDevType = 100;
      else
        tmp_parent->childDevType = 300;

      nextData->parent = tmp_parent;
      nextData->visited = 1;
      done = 1;
    }
    UmMutexUnlock(parent_mutex);
  } while (!done);
}

void BstBenchmark::InitializeNodes(Node *data, uint32_t num_nodes, int seed) {
  Node *tmp_node;
  int64_t val;

  srand(seed);
  for (size_t i = 0; i < num_nodes; i++) {
    tmp_node = &(data[i]);

    val = (((rand() & 255) << 8 | (rand() & 255)) << 8 | (rand() & 255)) << 7 |
          (rand() & 127);

    tmp_node->value = val;
    tmp_node->left = NULL;
    tmp_node->right = NULL;
    tmp_node->parent = NULL;
    tmp_node->visited = 0;
    tmp_node->childDevType = -1;

    UmMutexInit(&tmp_node->mutex_node, UM_MUTEX_UNLOCK);
  }
}

Node *BstBenchmark::MakeBinaryTree(uint32_t num_nodes, Node *inroot) {
  Node *root = NULL;
  Node *data;
  Node *nextData;

  if (NULL != inroot) {
    /* allocate first node to root */
    data = reinterpret_cast<Node *>(inroot);
    nextData = data;
    root = nextData;

    /* iterative tree insert */
    for (size_t i = 1; i < num_nodes; ++i) {
      nextData = nextData + 1;

      InsertNode(nextData, root);
    }
  }

  return root;
}

uint32_t BstBenchmark::CountNodes(Node *root) {
  uint32_t count = 0;
  if (root) count = 1;

  if (root->left) count += CountNodes(root->left);

  if (root->right) count += CountNodes(root->right);

  return count;
}

void BstBenchmark::Initialize() {
  host_nodes_ =
      static_cast<int>(1.0 * num_insert_ * (1.0 * host_percentage_ / 100));
  device_nodes_ = num_insert_ - host_nodes_;
  total_nodes_ = num_insert_ + init_tree_insert_;
  printf("Host nodes are %d \n", host_nodes_);
  printf("Device nodes are %d \n", device_nodes_);
}

void BstBenchmark::Verify() {}

void BstBenchmark::Summarize() {}

void BstBenchmark::Cleanup() {}
