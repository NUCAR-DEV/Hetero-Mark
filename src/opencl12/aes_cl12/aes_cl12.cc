/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
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
 * Advanced Encryption Code With OpenCL 1.2
 *
 * It takes a plain text or hex file and encrypts it with a given key
 *
 */

#include "include/aes_cl12.h"

#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <memory>
#include <sstream>

AES::AES() {
  runtime  = clRuntime::getInstance();
  file     = clFile::getInstance();

  platform = runtime->getPlatformID();
  device   = runtime->getDevice();
  context  = runtime->getContext();
  cmdQueue = runtime->getCmdQueue(0);

  // Init
  RUNNING_THREADS = MAX_WORK_ITEMS * BASIC_UNIT;
  expanded_key[60] = { 0x00 };
  hexMode = 1;
}

AES::~AES() {
  FreeKernel();
}

int AES::InitFiles() {
  //Note: fp is a struct defined in the header and set using
  //setInitialParameters(FilePackage filepackage)

  // The first argument is the hexMode
  if (strcmp(fp.mode, "h") == 0) {
    hexMode = true;
  } else if (strcmp(fp.mode, "a") == 0) {
    hexMode = false;
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
  append_str << "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n"
             << "#define Nb 4\n"
             << "#define Nr 14\n"
             << "define Nk 8\n"
             << "\n__constant uint eK[60]={";

  for (int i = 0; i < 60; ++i) {
    append_str << "0x" << std::hex << expanded_key[i];
    if (i != 59)
      append_str << ",";
  }
  append_str << "};\n";

  // Open kernel file
  file->open("aes_cl12_Kernels.cl");

  // Append kernel source code
  append_str << file->getSource();

  // Create program
  std::string s = append_str.str();
  const char *source = s.c_str();
  program = clCreateProgramWithSource(context, 1,
                                     (const char **)&source, NULL, &err);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0x10000,
                          buf,
                          NULL);
    printf("Build info:\n%s\n", buf);
    exit(-1);
  }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");

  kernel = clCreateKernel(program, "CLRunnerntrl", &err);
  checkOpenCLErrors(err, "Failed to create AES kernel\n");
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

uint32_t AES::RotateWord(uint32_t word) {
  // Unions allow the 32-bit word to be operated on without
  // any memory copies or transformations
  union {
    uint8_t bytes[4];
    uint32_t word;
  } subWord  __attribute__((aligned));

  // Note: The word is stored backwards, that is why the index
  // starts at 3 and goes to 0x00
  subWord.word = word;

  uint8_t B0 = subWord.bytes[3];
  uint8_t B1 = subWord.bytes[2];
  uint8_t B2 = subWord.bytes[1];
  uint8_t B3 = subWord.bytes[0];

  subWord.bytes[3] = B1;  // 0
  subWord.bytes[2] = B2;  // 1
  subWord.bytes[1] = B3;  // 2
  subWord.bytes[0] = B0;  // 3

  return subWord.word;
}

uint32_t AES::SubWord(uint32_t word) {
  union {
    uint32_t word;
    uint8_t bytes[4];
  } subWord  __attribute__((aligned));

  subWord.word = word;

  subWord.bytes[3] = s[subWord.bytes[3]];
  subWord.bytes[2] = s[subWord.bytes[2]];
  subWord.bytes[1] = s[subWord.bytes[1]];
  subWord.bytes[0] = s[subWord.bytes[0]];

  return subWord.word;
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
    univar[i].bytes[3] = pk[i*4];
    univar[i].bytes[2] = pk[i*4+1];
    univar[i].bytes[1] = pk[i*4+2];
    univar[i].bytes[0] = pk[i*4+3];
  }

  for (i = Nk; i < Nb*(Nr+1); i++) {
    temp.word = univar[i-1].word;
    if (i % Nk == 0) {
      temp.word = (SubWord(RotateWord(temp.word)));
      temp.bytes[3] = temp.bytes[3] ^ (Rcon[i/Nk]);
    } else if (Nk > 6 && i % Nk == 4) {
      temp.word = SubWord(temp.word);
    }
    if (i-4 % Nk == 0)
      temp.word = SubWord(temp.word);

    univar[i].word = univar[i-Nk].word ^ temp.word;
  }

  // Copy from the buffer into the variable
  for (i = 0; i < 60; i++)
    expanded_key[i] = univar[i].word;
}

void AES::InitKeys() {
  // Read the private key in
  for (int i = 0; i < 32; i++)
    int res = fscanf(keyfile, "%02x", (int *)&key[i]);
  // res is simply a variable to supress the "ignored"
  // results warning
  // Expand key
  KeyExpansion(key);

  //Copy key to GPU
  cl_int err;
  clKey = clCreateBuffer(context,
   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
   60*sizeof(uint32_t), 
   expanded_key, 
   &err);
  if (err != CL_SUCCESS)
    printf("Error copying key to GPU");
}

void AES::Run() {
  cl_int err;

  InitFiles();
  InitKeys();
  InitKernel();

  int ch    = 0;  // The buffer for the data read in using ASCII/binary mode
  int spawn = 0;  // The number of compute units that will be enqueued per cycle
  int end   = 1;  // Changed to 0 when the end of the file is reached,
                  // terminates the infinite loop
  unsigned int currentOffset = 0;  // Data is linearly dimensionalized
  uint8_t states[16 * RUNNING_THREADS];
  while (end) {
    spawn = 0;
    // Dispatch many control threads that will report back to main
    // (for now 5x) - 1 worker per state
    for (int i = 0; i < RUNNING_THREADS; i++) {
      currentOffset = i*16;
      spawn++;
      for (int ix = 0; ix < 16; ix++) {
        if (hexMode == 1) {
          if (fscanf(infile, "%02x",
                     (unsigned int *)&states[currentOffset+ix]) != EOF) {
          } else {
            if (ix > 0) {
              for (int ixx = ix; ixx < 16; ixx++) {
                states[currentOffset+ixx] = 0x00;
              }
            } else { spawn--; }
            i = RUNNING_THREADS + 1;
            end = 0;
            break;
          }
        } else {
          ch = getc(infile);
          if (ch != EOF) { states[currentOffset+ix] = ch;
          } else {
            if (ix > 0) { for (int ixx = ix; ixx < 16; ixx++) {
                states[currentOffset+ixx] = 0x00;
              }
            } else { spawn--; }
            i = RUNNING_THREADS + 1;
            end = 0;
            break;
          }
        }
      }
    }
    if (spawn == 0) { break; }
    // arrange data correctly
    for (int i = 0; i < spawn; i++) {
      currentOffset = i*16;
      uint8_t temp[16];
      memcpy(&temp[0], &states[currentOffset], sizeof(uint8_t));
      memcpy(&temp[4], &states[currentOffset+1], sizeof(uint8_t));
      memcpy(&temp[8], &states[currentOffset+2], sizeof(uint8_t));
      memcpy(&temp[12], &states[currentOffset+3], sizeof(uint8_t));
      memcpy(&temp[1], &states[currentOffset+4], sizeof(uint8_t));
      memcpy(&temp[5], &states[currentOffset+5], sizeof(uint8_t));
      memcpy(&temp[9], &states[currentOffset+6], sizeof(uint8_t));
      memcpy(&temp[13], &states[currentOffset+7], sizeof(uint8_t));
      memcpy(&temp[2], &states[currentOffset+8], sizeof(uint8_t));
      memcpy(&temp[6], &states[currentOffset+9], sizeof(uint8_t));
      memcpy(&temp[10], &states[currentOffset+10], sizeof(uint8_t));
      memcpy(&temp[14], &states[currentOffset+11], sizeof(uint8_t));
      memcpy(&temp[3], &states[currentOffset+12], sizeof(uint8_t));
      memcpy(&temp[7], &states[currentOffset+13], sizeof(uint8_t));
      memcpy(&temp[11], &states[currentOffset+14], sizeof(uint8_t));
      memcpy(&temp[15], &states[currentOffset+15], sizeof(uint8_t));
      for (int c = 0; c < 16; c++) {
        memcpy(&states[currentOffset+c], &temp[c], sizeof(uint8_t));
      }
    }
    // Set data for workers----------

    cl_int status;

    dev_states = clCreateBuffer(context, CL_MEM_READ_WRITE | \
               CL_MEM_COPY_HOST_PTR, 16*spawn*sizeof(uint8_t), states, &status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_states);
    checkOpenCLErrors(status, "clSetKernelArg(data)\n");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clKey);
    checkOpenCLErrors(status, "clSetKernelArg(key)\n");

    // Calculations to optimize the execution of the kernel
    size_t local_ws;
    const size_t global_ws = spawn;
    if (spawn < BASIC_UNIT) { local_ws = 1;
    } else if (spawn % BASIC_UNIT > 0) { local_ws = (spawn/BASIC_UNIT) + 1;
    } else { local_ws = (spawn/BASIC_UNIT); }

    cl_event event;
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, \
                                     &global_ws, &local_ws, 0, NULL, &event);
    checkOpenCLErrors(status, "clEnqueueNDRangeKernel\n");

    clFinish(cmdQueue);

    status = clEnqueueReadBuffer(cmdQueue, dev_states, CL_TRUE, 0, \
                            16*spawn*sizeof(uint8_t), &states, 0, NULL, NULL);

    checkOpenCLErrors(status, "clEnqueueReadBuffer\n");

    clReleaseMemObject(dev_states);

    for (int i = 0; i < spawn; i++) {
      currentOffset = i*16;
      for (int ix = 0; ix < 4; ix++) {
        char hex[3];
        sprintf(hex, "%02x", states[currentOffset+ix]);
        for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
        sprintf(hex, "%02x", states[currentOffset+ix+4]);
        for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
        sprintf(hex, "%02x", states[currentOffset+ix+8]);
        for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
        sprintf(hex, "%02x", states[currentOffset+ix+12]);
        for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
      }
    }
  }  // while

  fflush(outfile);
}
