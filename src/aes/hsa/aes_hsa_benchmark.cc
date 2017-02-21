/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Carter McCardwell (carter@mccardwell.net, cmccardw@ece.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 */

#include "src/aes/hsa/aes_hsa_benchmark.h"
#include <cstdio>
#include <cstring>
#include "src/aes/hsa/kernels.h"

void AesHsaBenchmark::Initialize() {
  AesBenchmark::Initialize();
  Encrypt_init(0);
}

void AesHsaBenchmark::Run() {
  ExpandKey();
  uint8_t *ciphertext = reinterpret_cast<uint8_t *>(
      malloc_global(text_length_));
  uint32_t *expanded_key = reinterpret_cast<uint32_t *>(
      malloc_global(kExpandedKeyLengthInBytes));
  uint8_t *d_s = reinterpret_cast<uint8_t *>(
      malloc_global(256));
  memcpy(ciphertext, ciphertext_, text_length_);
  memcpy(expanded_key, expanded_key_, kExpandedKeyLengthInBytes);
  memcpy(d_s, s, 256);

  SNK_INIT_LPARM(lparm, 0);
  int num_blocks = text_length_ / 16;
  lparm->gdims[0] = num_blocks;
  lparm->ldims[0] = 64 < num_blocks ? 64 : num_blocks;

  Encrypt(ciphertext, expanded_key, d_s, lparm);
    
  memcpy(ciphertext_, ciphertext, text_length_);

  free_global(ciphertext);
  free_global(expanded_key);
  free_global(d_s);
}
