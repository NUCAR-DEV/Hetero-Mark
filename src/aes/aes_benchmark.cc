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

#include "src/aes/aes_benchmark.h"
#include <inttypes.h>
#include <cstdio>
#include <cstring>

void AesBenchmark::Initialize() {
  LoadPlaintext();

  LoadKey();

  InitiateCiphertext(&ciphertext_);
}

void AesBenchmark::LoadPlaintext() {
  FILE *input_file = fopen(input_file_name_.c_str(), "r");
  if (!input_file) {
    fprintf(stderr, "Fail to open input file.\n");
    exit(1);
  }

  // Get the size of file
  fseek(input_file, 0L, SEEK_END);
  uint64_t file_size = ftell(input_file);
  fseek(input_file, 0L, SEEK_SET);

  // 2 char per hex number
  text_length_ = file_size / 2;
  plaintext_ = reinterpret_cast<uint8_t *>(malloc(text_length_));

  uint64_t index = 0;
  unsigned int byte;
  while (fscanf(input_file, "%02x", &byte) == 1) {
    plaintext_[index] = static_cast<uint8_t>(byte);
    index++;
  }

  fclose(input_file);
}

void AesBenchmark::LoadKey() {
  FILE *key_file = fopen(key_file_name_.c_str(), "r");
  if (!key_file) {
    fprintf(stderr, "Fail to open key file.\n");
    exit(1);
  }

  unsigned int byte;
  for (int i = 0; i < kKeyLengthInBytes; i++) {
    if (!fscanf(key_file, "%02x", &byte)) {
      fprintf(stderr, "Error in key file.\n");
      exit(1);
    }
    key_[i] = static_cast<uint8_t>(byte);
  }

  fclose(key_file);
}

void AesBenchmark::InitiateCiphertext(uint8_t **ciphertext) {
  *ciphertext = reinterpret_cast<uint8_t *>(malloc(text_length_));
  memcpy(*ciphertext, plaintext_, text_length_);
}

void AesBenchmark::DumpText(uint8_t *text) {
  printf("Text: ");
  for (uint64_t i = 0; i < text_length_; i++) {
    printf("%02x ", text[i]);
  }
  printf("\n");
}

void AesBenchmark::DumpKey() {
  printf("Key: ");
  for (int i = 0; i < kKeyLengthInBytes; i++) {
    printf("%02x ", key_[i]);
  }
  printf("\n");
}

void AesBenchmark::DumpExpandedKey() {
  printf("Expanded key: ");
  for (int i = 0; i < kExpandedKeyLengthInWords; i++) {
    printf("\nword[%d]: %08x ", i, expanded_key_[i]);
  }
  printf("\n");
}

void AesBenchmark::Verify() {
  uint8_t *ciphertext_cpu = nullptr;
  InitiateCiphertext(&ciphertext_cpu);
  // memcpy(ciphertext_cpu, plaintext_, text_length_);

  ExpandKey();

  uint8_t *encrypt_ptr = ciphertext_cpu;
  while (encrypt_ptr < ciphertext_cpu + text_length_) {
    uint8_t *state;
    state = encrypt_ptr;

    AddRoundKeyCpu(state, 0);

    for (int i = 1; i < kNumRounds; i++) {
      SubBytesCpu(state);
      ShiftRowsCpu(state);
      MixColumnsCpu(state);
      AddRoundKeyCpu(state, i * kBlockSizeInWords);
    }

    SubBytesCpu(state);
    ShiftRowsCpu(state);
    AddRoundKeyCpu(state, 14 * kBlockSizeInWords);

    encrypt_ptr += kBlockSizeInBytes;
  }

  bool passed = true;
  for (uint64_t i = 0; i < text_length_; i++) {
    if (ciphertext_cpu[i] != ciphertext_[i]) {
      passed = false;
      printf("Position: %" PRIu64 ", expected to be 0x%02x, but get 0x%02x\n",
             i, ciphertext_cpu[i], ciphertext_[i]);
      exit(-1);
      return;
    }
  }

  if (passed) {
    printf("Passed. %" PRIu64 " bytes encrypted.\n", text_length_);
  }
}

void AesBenchmark::AddRoundKeyCpu(uint8_t *state, uint8_t offset) {
  for (int i = 0; i < kBlockSizeInWords; i++) {
    uint32_t state_word = BytesToWord(state + 4 * i);
    uint32_t after_xor_key = state_word ^ expanded_key_[offset + i];
    WordToBytes(after_xor_key, state + 4 * i);
  }
}

void AesBenchmark::SubBytesCpu(uint8_t *state) {
  for (int i = 0; i < kBlockSizeInBytes; i++) {
    state[i] = s[state[i]];
  }
}

void AesBenchmark::ShiftRowsCpu(uint8_t *state) {
  for (int i = 0; i < 4; i++) {
    uint8_t bytes[4];
    bytes[0] = state[i];
    bytes[1] = state[i + 4];
    bytes[2] = state[i + 8];
    bytes[3] = state[i + 12];

    uint32_t word = BytesToWord(bytes);
    for (int j = 0; j < i; j++) {
      word = RotateWord(word);
    }
    WordToBytes(word, bytes);

    state[i] = bytes[0];
    state[i + 4] = bytes[1];
    state[i + 8] = bytes[2];
    state[i + 12] = bytes[3];
  }
}

void AesBenchmark::MixColumnsCpu(uint8_t *state) {
  for (int i = 0; i < kBlockSizeInWords; i++) {
    MixColumnsOneWord(state + 4 * i);
  }
}

void AesBenchmark::MixColumnsOneWord(uint8_t *word) {
  uint8_t a[4];
  uint8_t b[4];
  uint8_t high_bit;
  for (int i = 0; i < 4; i++) {
    a[i] = word[i];
    high_bit = word[i] & 0x80;
    b[i] = word[i] << 1;
    if (high_bit == 0x80) {
      b[i] ^= 0x1b;
    }
  }
  word[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
  word[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
  word[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
  word[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];
}

void AesBenchmark::ExpandKey() {
  cpu_gpu_logger_->CPUOn();

  for (int i = 0; i < kKeyLengthInWords; i++) {
    expanded_key_[i] = BytesToWord(key_ + 4 * i);
  }

  uint32_t temp;
  for (int i = kKeyLengthInWords; i < kExpandedKeyLengthInWords; i++) {
    temp = expanded_key_[i - 1];

    if (i % kKeyLengthInWords == 0) {
      uint32_t after_rotate_word = RotateWord(temp);
      uint32_t after_sub_word = SubWord(after_rotate_word);
      uint32_t rcon_word = Rcon[i / kKeyLengthInWords] << 24;
      temp = after_sub_word ^ rcon_word;
    } else if (i % kKeyLengthInWords == 4) {
      temp = SubWord(temp);
    }
    expanded_key_[i] = expanded_key_[i - kKeyLengthInWords] ^ temp;
  }

  cpu_gpu_logger_->CPUOff();
}

void AesBenchmark::WordToBytes(uint32_t word, uint8_t *bytes) {
  bytes[0] = (word & 0xff000000) >> 24;
  bytes[1] = (word & 0x00ff0000) >> 16;
  bytes[2] = (word & 0x0000ff00) >> 8;
  bytes[3] = (word & 0x000000ff) >> 0;
}

uint32_t AesBenchmark::BytesToWord(uint8_t *bytes) {
  return (bytes[0] << 24) + (bytes[1] << 16) + (bytes[2] << 8) + bytes[3];
}

uint32_t AesBenchmark::RotateWord(uint32_t word) {
  uint32_t after_rotation;
  uint8_t bytes[5];
  WordToBytes(word, bytes);
  bytes[4] = bytes[0];

  after_rotation = BytesToWord(bytes + 1);
  return after_rotation;
}

uint32_t AesBenchmark::SubWord(uint32_t word) {
  uint32_t after_subword;
  uint8_t bytes[4];

  WordToBytes(word, bytes);

  for (int i = 0; i < 4; i++) {
    bytes[i] = s[bytes[i]];
  }

  after_subword = BytesToWord(bytes);
  return after_subword;
}

void AesBenchmark::Summarize() {
  printf("Plaintext: ");
  DumpText(plaintext_);

  DumpKey();

  printf("Ciphertext: ");
  DumpText(ciphertext_);
}

void AesBenchmark::Cleanup() {
  free(plaintext_);
  free(ciphertext_);
}
