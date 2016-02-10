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
 *
 * Advanced Encryption Code With OpenCL 2.0
 *
 * It takes a plain text or hex file and encrypts it with a given key
 *
 */
#include "include/aes_cl20.h"

#include <string.h>
#include <inttypes.h>
#include <memory>
#include <sstream>

#define ENABLE_PROFILE 1

#if ENABLE_PROFILE
#define clEnqueueNDRangeKernel clTimeNDRangeKernel
#endif

AES::AES() {
  runtime = clRuntime::getInstance();
  file = clFile::getInstance();

  platform = runtime->getPlatformID();
  device = runtime->getDevice();
  context = runtime->getContext();
  cmd_queue = runtime->getCmdQueue(0);

  // Init
  MAXIMUM_MEMORY_ALLOCATION = 1;
  expanded_key[59] = {0x00};
  hex_mode = 1;
}

AES::~AES() {
  FreeKernel();
  FreeBuffer();
}

void AES::InitFiles() {
  // Note: fp is a struct defined in the header and set using
  // setInitialParameters(FilePackage filepackage)

  // The first argument is the hex_mode
  if (strcmp(fp.mode, "h") == 0) {
    hex_mode = true;
  } else if (strcmp(fp.mode, "a") == 0) {
    hex_mode = false;
  } else {
    printf("error: first argument must be \'a\' for ASCII interpretation");
    printf(" or \'h\' for hex interpretation\n");
    exit(-1);
  }

  // The second argument is the infile
  infile = fopen(fp.in, "r");
  if (!infile) {
    printf("error_in\n");
    exit(-1);
  }

  // The third argument is the keyfile, it must be in hex
  // and broken into two charactor parts (eg. AA BB CC ...)
  keyfile = fopen(fp.key, "rb");
  if (!keyfile) {
    printf("error_key\n");
    exit(-1);
  }

  // The outfile, the encrypted results will be written here
  outfile = fopen(fp.out, "w");
  if (!outfile) {
    printf("error (permission error: run with sudo or");
    printf(" in directory the user owns)\n");
    exit(-1);
  }
}

void AES::InitKernel() {
  cl_int err;

  // Need to patch kernel source
  std::stringstream append_str;
  append_str
      << "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n"
      << "#define Nb 4\n"
      << "#define Nr 14\n"
      << "#define Nk 8\n"
      << "\n__constant uint eK[60]={";
  for (int i = 0; i < 60; ++i) {
    append_str << "0x" << std::hex << expanded_key[i];
    if (i != 59) append_str << ",";
  }
  append_str << "};\n";

  // Open kernel file
  file->open("aes_Kernels.cl");

  // Append kernel source code
  append_str << file->getSource();

  // Create program
  std::string s = append_str.str();
  const char *source = s.c_str();
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, buf,
                          NULL);
    printf("Build info:\n%s\n", buf);
    exit(-1);
  }

  // Create program with OpenCL 2.0 support
  err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");

  kernel = clCreateKernel(program, "CLRunner", &err);
  checkOpenCLErrors(err, "Failed to create AES kernel\n");
}

void AES::InitBuffer() {
  MAXIMUM_MEMORY_ALLOCATION = runtime->getNumComputeUnit();
  svm_ptr = reinterpret_cast<uint8_t *>(
      clSVMAlloc(context, CL_MEM_READ_WRITE,
                 sizeof(uint8_t) * 16 * MAXIMUM_MEMORY_ALLOCATION, 0));
  if (!svm_ptr) std::cout << "Failed to clSVMAlloc svm_ptr" << std::endl;
}

void AES::FreeFiles() {
  fclose(infile);
  fclose(keyfile);
  fclose(outfile);
}

void AES::FreeKernel() {
  cl_int err;
  err = clReleaseKernel(kernel);
  checkOpenCLErrors(err, "Failed to release kernel");
}

void AES::FreeBuffer() {
  if (svm_ptr) clSVMFree(context, svm_ptr);
}

uint32_t AES::RotateWord(uint32_t word) {
  // Unions allow the 32-bit word to be operated on without
  // any memory copies or transformations
  union {
    uint8_t bytes[4];
    uint32_t word;
  } sub_word __attribute__((aligned));
  // Note: The word is stored backwards, that is why the index
  // starts at 3 and goes to 0x00
  sub_word.word = word;

  uint8_t B0 = sub_word.bytes[3];
  uint8_t B1 = sub_word.bytes[2];
  uint8_t B2 = sub_word.bytes[1];
  uint8_t B3 = sub_word.bytes[0];

  sub_word.bytes[3] = B1;  // 0
  sub_word.bytes[2] = B2;  // 1
  sub_word.bytes[1] = B3;  // 2
  sub_word.bytes[0] = B0;  // 3

  return sub_word.word;
}

uint32_t AES::SubWord(uint32_t word) {
  union {
    uint32_t word;
    uint8_t bytes[4];
  } sub_word __attribute__((aligned));

  sub_word.word = word;

  sub_word.bytes[3] = s[sub_word.bytes[3]];
  sub_word.bytes[2] = s[sub_word.bytes[2]];
  sub_word.bytes[1] = s[sub_word.bytes[1]];
  sub_word.bytes[0] = s[sub_word.bytes[0]];

  return sub_word.word;
}

void AES::KeyExpansion(uint8_t *pk) {
  int i = 0;

  // Temp union will hold the word that is being processed
  union {
    uint8_t bytes[4];
    uint32_t word;
  } temp __attribute__((aligned));

  // Univar is the buffer that will hold the expanded key, it
  // is loaded in 8-bit parts which is why the union is necessary
  union {
    uint8_t bytes[4];
    uint32_t word;
  } univar[60] __attribute__((aligned));

  for (i = 0; i < Nk; i++) {
    univar[i].bytes[3] = pk[i * 4];
    univar[i].bytes[2] = pk[i * 4 + 1];
    univar[i].bytes[1] = pk[i * 4 + 2];
    univar[i].bytes[0] = pk[i * 4 + 3];
  }

  for (i = Nk; i < Nb * (Nr + 1); i++) {
    temp.word = univar[i - 1].word;
    if (i % Nk == 0) {
      temp.word = (SubWord(RotateWord(temp.word)));
      temp.bytes[3] = temp.bytes[3] ^ (Rcon[i / Nk]);
    } else if (Nk > 6 && i % Nk == 4) {
      temp.word = SubWord(temp.word);
    }

    if (i - 4 % Nk == 0) {
      temp.word = SubWord(temp.word);
    }

    univar[i].word = univar[i - Nk].word ^ temp.word;
  }

  // Copy from the buffer into the variable
  for (i = 0; i < 60; i++) expanded_key[i] = univar[i].word;
}

void AES::InitKeys() {
  // Read the private key in
  for (int i = 0; i < 32; i++)
    if (fscanf(keyfile, "%02x", reinterpret_cast<int *>(&key[i]))) {
    }
  // If statement to suppress the "ignored"
  // result warning
  // Expand key
  KeyExpansion(key);
}

void AES::Initialize() {
  InitFiles();
  InitKeys();
  InitKernel();
  InitBuffer();
}

void AES::Run() {
  cl_int err;

  int ch = 0;     // The buffer for the data read in using ASCII/binary mode
  int spawn = 0;  // The number of compute units that will be enqueued per cycle
  int end = 1;    // Changed to 0 when the end of the file is reached,
                  // terminates the infinite loop
  while (end) {
    // Map memory location in GPU <Coarse SVM>
    err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_WRITE, svm_ptr,
                          sizeof(uint8_t) * 16 * MAXIMUM_MEMORY_ALLOCATION, 0,
                          0, 0);
    checkOpenCLErrors(err, "Failed at clEnqueueSVMMap");

    spawn = 0;  // Each cycle reset the spawn to zero
    for (int i = 0; i < MAXIMUM_MEMORY_ALLOCATION; i++) {
      spawn++;  // Since another state is being filled,
                // another comptute unit will be required
      for (int ix = 0; ix < 16; ix++) {
        if (hex_mode) {  // If hex mode
          if (fscanf(infile, "%02x", (unsigned int *)&svm_ptr[(i * 16) + ix]) !=
              EOF) {
          } else {  // Reads in hex
            // If the end of the file is reached,
            // fill the rest of the state with 0x00
            if (ix > 0) {
              for (int ixx = ix; ixx < 16; ixx++)
                svm_ptr[(i * 16) + ixx] = 0x00;
            } else {
              // If the end of the file is reached before even partially
              // filling a state, do not process it.
              spawn--;
            }
            i = MAXIMUM_MEMORY_ALLOCATION + 1;  // Break middle loop
            end = 0;                            // Break infinite loop

            break;
          }
        } else {  // If ASCII-binary mode
          ch = getc(infile);
          if (ch != EOF) {
            svm_ptr[(i * 16) + ix] = ch;
          } else {
            if (ix > 0) {
              for (int ixx = ix; ixx < 16; ixx++) {
                svm_ptr[(i * 16) + ixx] = 0x00;
              }
            } else {
              spawn--;
            }

            i = MAXIMUM_MEMORY_ALLOCATION + 1;
            end = 0;

            break;
          }
        }
      }
    }
    // Arrange data correctly, the AES algoritm requres the state to be
    // orientated in a specific way (explained in the specification)
    for (int i = 0; i < spawn; i++) {
      uint8_t temp[16];
      memcpy(&temp[0], &svm_ptr[(i * 16) + 0], sizeof(uint8_t));
      memcpy(&temp[4], &svm_ptr[(i * 16) + 1], sizeof(uint8_t));
      memcpy(&temp[8], &svm_ptr[(i * 16) + 2], sizeof(uint8_t));
      memcpy(&temp[12], &svm_ptr[(i * 16) + 3], sizeof(uint8_t));
      memcpy(&temp[1], &svm_ptr[(i * 16) + 4], sizeof(uint8_t));
      memcpy(&temp[5], &svm_ptr[(i * 16) + 5], sizeof(uint8_t));
      memcpy(&temp[9], &svm_ptr[(i * 16) + 6], sizeof(uint8_t));
      memcpy(&temp[13], &svm_ptr[(i * 16) + 7], sizeof(uint8_t));
      memcpy(&temp[2], &svm_ptr[(i * 16) + 8], sizeof(uint8_t));
      memcpy(&temp[6], &svm_ptr[(i * 16) + 9], sizeof(uint8_t));
      memcpy(&temp[10], &svm_ptr[(i * 16) + 10], sizeof(uint8_t));
      memcpy(&temp[14], &svm_ptr[(i * 16) + 11], sizeof(uint8_t));
      memcpy(&temp[3], &svm_ptr[(i * 16) + 12], sizeof(uint8_t));
      memcpy(&temp[7], &svm_ptr[(i * 16) + 13], sizeof(uint8_t));
      memcpy(&temp[11], &svm_ptr[(i * 16) + 14], sizeof(uint8_t));
      memcpy(&temp[15], &svm_ptr[(i * 16) + 15], sizeof(uint8_t));

      for (int c = 0; c < 16; c++)
        memcpy(&svm_ptr[(i * 16) + c], &temp[c], sizeof(uint8_t));
    }

    // All data loaded, unmap the state memory before launching kernel
    cl_event unmap;
    err = clEnqueueSVMUnmap(cmd_queue, svm_ptr, 0, 0, &unmap);
    clWaitForEvents(1, &unmap);
    checkOpenCLErrors(err, "Failed to unmap SVM");

    // const size_t local_ws = 0;
    // For now, let the runtime decide on the local
    const size_t global_ws = spawn;  // One WU per state in this round

    err =
        clSetKernelArgSVMPointer(kernel, 0, reinterpret_cast<void *>(svm_ptr));
    // Specific function for SVM data
    checkOpenCLErrors(err, "Failed to set kernel SVM arg ");

    // Start the kernel
    err =
        clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, &global_ws, 0, 0, 0, 0);
    checkOpenCLErrors(err, "Failed to execute kernel");

    // Write processed data back to out
    err = clEnqueueSVMMap(cmd_queue, CL_TRUE, CL_MAP_READ, svm_ptr,
                          sizeof(uint8_t) * 16 * MAXIMUM_MEMORY_ALLOCATION, 0,
                          0, 0);
    checkOpenCLErrors(err, "Failed to map SVM ptr");

    // Writes the processed data to the outfile,
    // rearranging it back into a linear format
    for (int i = 0; i < spawn; i++) {
      for (int ix = 0; ix < 4; ix++) {
        char hex[3];
        snprintf(hex, sizeof(hex), "%02x", svm_ptr[(i * 16) + ix]);
        for (int i = 0; i < 3; i++) {
          putc(hex[i], outfile);
        }
        snprintf(hex, sizeof(hex), "%02x", svm_ptr[(i * 16) + ix + 4]);
        for (int i = 0; i < 3; i++) {
          putc(hex[i], outfile);
        }
        snprintf(hex, sizeof(hex), "%02x", svm_ptr[(i * 16) + ix + 8]);
        for (int i = 0; i < 3; i++) {
          putc(hex[i], outfile);
        }
        snprintf(hex, sizeof(hex), "%02x", svm_ptr[(i * 16) + ix + 12]);
        for (int i = 0; i < 3; i++) {
          putc(hex[i], outfile);
        }
      }
    }

    // Release the SVM buffer before the next cycle
    err = clEnqueueSVMUnmap(cmd_queue, svm_ptr, 0, 0, 0);
    checkOpenCLErrors(err, "Failed at unmap SVM ptr");
  }  // While

  fflush(outfile);
}
