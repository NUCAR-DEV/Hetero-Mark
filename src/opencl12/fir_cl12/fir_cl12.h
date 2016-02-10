/*
* Copyright (c) 2015 Northeastern University
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
*/

#ifndef FIR_H
#define FIR_H

#include "src/common/cl_util/cl_util.h"
#include "src/common/time_measurement/time_measurement.h"
#include "src/common/benchmark/benchmark.h"

using namespace clHelper;

class FIR : public Benchmark {
 private:
  clRuntime* runtime_;
  clFile* file_;

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_program program_;
  cl_command_queue cmd_queue_;
  cl_kernel fir_kernel_;

  cl_uint num_tap_ = 16;
  cl_uint num_data_ = 0;
  cl_uint num_total_data_ = 0;
  cl_uint num_blocks_ = 0;
  cl_float* input_ = NULL;
  cl_float* output_ = NULL;
  cl_float* coeff_ = NULL;
  cl_float* temp_output_ = NULL;
  cl_mem output_buffer_;
  cl_mem coeff_buffer_;
  cl_mem temp_output_buffer_;

  TimeMeasurement* timer;

  void InitializeCL();
  void InitializeData();
  void InitializeKernels();
  void InitializeBuffers();

 public:
  FIR(){};
  ~FIR(){};

  void SetInitialParameters(int num_data, int num_blocks) {
    this->num_blocks_ = num_blocks;
    this->num_data_ = num_data;
  }

  void SetTimer(TimeMeasurement* timer) { this->timer = timer; }

  void Initialize() override;
  void Run() override;
  void Verify() override;
  void Cleanup() override;
  void Summarize() override {}
};

#endif
