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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "src/bst/cuda/bst_cuda_benchmark.h"

__global__ void bst_cuda(void *tree_root, void *dev_start_node,
                         uint32_t gpu_nodes) {
  volatile um_mutex *tmp_mutex;
  Node *tmp_node, *tmp_parent, *new_node;

  Node *root = reinterpret_cast<Node *>(tree_root);
  Node *data = reinterpret_cast<Node *>(dev_start_node);

  int64_t flag;
  int64_t key;

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= gpu_nodes) return;

  flag = 0;
  tmp_node = root;

  tmp_parent = root;
  new_node = &(data[tid]);

  key = new_node->value;

  /* Search the parent node.*/
  /* Multiple work-items in the a work-group run this part. */
  while (tmp_node) {
    tmp_parent = tmp_node;
    flag = (key - (tmp_node->value));
    tmp_node = (flag < 0) ? tmp_node->left : tmp_node->right;
  }

  Node *child = tmp_node;
  int done = 0;
  tmp_mutex = &tmp_parent->mutex_node;
  int exFlag, expected;

  do {
    tmp_mutex = &tmp_parent->mutex_node;
    expected = UM_MUTEX_UNLOCK;

    exFlag = atomicCAS_system(reinterpret_cast<int *>(&tmp_mutex->count),
                              expected, 1);

    // If Parent node lock is successful
    if (exFlag == UM_MUTEX_UNLOCK) {
      child = (flag < 0) ? tmp_parent->left : tmp_parent->right;
      if (child) {
        // Parent node has been updated since last check. Get the new parent and
        // iterate again
        tmp_parent = child;
      } else {
        // Insert the node
        tmp_parent->left = (flag < 0) ? new_node : tmp_parent->left;
        tmp_parent->right = (flag >= 0) ? new_node : tmp_parent->right;

        // Whether device only insert (childDevType=200) or both host and device
        // (childDevType=300)
        if (tmp_parent->childDevType == -1 || tmp_parent->childDevType == 200)
          tmp_parent->childDevType = 200;
        else
          tmp_parent->childDevType = 300;

        new_node->parent = tmp_parent;
        new_node->visited = 1;
        done = 1;
      }

      expected = UM_MUTEX_LOCK;

      atomicCAS_system(reinterpret_cast<int *>(&tmp_mutex->count), expected, 0);
    }

    __threadfence_system();
  } while (!done);
}

void BstCudaBenchmark::Initialize() {
  BstBenchmark::Initialize();
  cudaMallocManaged(&tree_buffer_, sizeof(Node) * total_nodes_);
  InitializeNodes(tree_buffer_, total_nodes_, seed_);
  root_ = MakeBinaryTree(init_tree_insert_, tree_buffer_);
}

void BstCudaBenchmark::Run() {
  dim3 block_size(64);
  dim3 grid_size((device_nodes_ + 64 - 1) / 64);

  printf("Grid size is %d \n", grid_size.x);
  uint32_t device_start_node = init_tree_insert_ + host_nodes_;

  printf("Device start node is %d \n", device_start_node);
  bst_cuda<<<grid_size, block_size>>>(
      reinterpret_cast<void *>(tree_buffer_),
      reinterpret_cast<void *>(tree_buffer_ + device_start_node),
      device_nodes_);

  uint32_t offset = 0;
  printf("Offset is %d \n", offset);
  for (int64_t k = 0; k < host_nodes_; k++) {
    InsertNode(&(tree_buffer_[init_tree_insert_ + offset + k]), root_);
  }

  cudaDeviceSynchronize();
  printf("Gpu done \n");
  uint32_t actual_nodes = CountNodes(tree_buffer_);

  printf("Number of actual nodes are %d \n", actual_nodes);
  printf("Number of total nodes are %d \n", total_nodes_);
}

void BstCudaBenchmark::Cleanup() { BstBenchmark::Cleanup(); }
