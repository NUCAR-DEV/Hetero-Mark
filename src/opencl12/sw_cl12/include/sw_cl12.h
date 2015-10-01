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
 * Shallow water Physics simulation engine
 *
 */

#ifndef SRC_OPENCL12_SW_CL12_SW_CL12_H_
#define SRC_OPENCL12_SW_CL12_SW_CL12_H_

#include "src/common/cl_util/cl_util.h"
#include "src/common/benchmark/benchmark.h"

using clHelper::clRuntime;
using clHelper::clFile;
using clHelper::clTimeNDRangeKernel;

class ShallowWater : public Benchmark {
  clRuntime *runtime_;
  clFile *file_;

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue cmdQueue_;

  cl_program program_;
  cl_kernel kernel_sw_init_psi_p_;
  cl_kernel kernel_sw_init_velocities_;
  cl_kernel kernel_sw_compute0_;
  cl_kernel kernel_sw_update0_;
  cl_kernel kernel_sw_compute1_;
  cl_kernel kernel_sw_update1_;
  cl_kernel kernel_sw_time_smooth_;

  // Size
  unsigned m_;
  unsigned n_;
  unsigned m_len_;
  unsigned n_len_;
  unsigned itmax_;

  // Params
  double dt_, tdt_, dx_, dy_, a_, alpha_, el_, pi_;
  double tpi_, di_, dj_, pcf_;
  double tdts8_, tdtsdx_, tdtsdy_, fsdx_, fsdy_;

  // OpenCL 1.2 style buffers
  cl_mem u_curr_;
  cl_mem u_next_;

  cl_mem v_curr_;
  cl_mem v_next_;

  cl_mem p_curr_;
  cl_mem p_next_;

  cl_mem u_;
  cl_mem v_;
  cl_mem p_;

  cl_mem cu_;
  cl_mem cv_;

  cl_mem z_;
  cl_mem h_;
  cl_mem psi_;

  // Initialize
  void InitKernel();
  void InitBuffer();
  void InitPsiP();
  void InitVelocities();

  // Cleanup
  void FreeKernel();
  void FreeBuffer();

  // Run
  void Compute0();
  void PeriodicUpdate0();
  void Compute1();
  void PeriodicUpdate1();
  void TimeSmooth(int ncycle);

 public:
  explicit ShallowWater(unsigned m = 2048, unsigned n = 2048);
  ~ShallowWater();

  void Initialize() override;
  void Run() override;
  void Verify() override {}
  void Cleanup() override;
  void Summarize() override {}
};

#endif  // SRC_OPENCL12_SW_CL12_SW_CL12_H_
