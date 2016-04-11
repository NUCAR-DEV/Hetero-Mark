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
 */

#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include "src/sw/cl12/sw_cl12_benchmark.h"

void SwCl12Benchmark::Initialize() {
  SwBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeParams();
  InitializeKernels();
  InitializeBuffers();
  InitializeData();
}

void SwCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  // Create kernels
  kernel_sw_init_psi_p_ = clCreateKernel(program_, "sw_init_psi_p", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_psi_p_");

  kernel_sw_init_velocities_ =
      clCreateKernel(program_, "sw_init_velocities", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_velocities_");

  kernel_sw_compute0_ = clCreateKernel(program_, "sw_compute0", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_compute0_");

  kernel_sw_update0_ = clCreateKernel(program_, "sw_update0", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_update0_");

  kernel_sw_compute1_ = clCreateKernel(program_, "sw_compute1", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_compute1_");

  kernel_sw_update1_ = clCreateKernel(program_, "sw_update1", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_update1_");

  kernel_sw_time_smooth_ = clCreateKernel(program_, "sw_time_smooth", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_time_smooth_");
}

void SwCl12Benchmark::InitializeBuffers() {
  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

  cl_int err;
  u_curr_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u_curr_");
  u_next_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u_next_");
  v_curr_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v_curr_");
  v_next_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v_next_");
  p_curr_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p_curr_");
  p_next_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p_next_");
  u_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u_");
  v_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v_");
  p_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p_");
  cu_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: cu_");
  cv_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: cv_");
  z_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: z_");
  h_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: h_");
  psi_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: psi_");
}

void SwCl12Benchmark::InitializeParams() {
  m_len_ = m_ + 1;
  n_len_ = n_ + 1;
  itmax_ = 250;
  dt_ = 90.;
  tdt_ = dt_;
  dx_ = 100000.;
  dy_ = 100000.;
  a_ = 1000000.;
  alpha_ = .001;
  el_ = n_ * dx_;
  pi_ = 4. * atanf(1.);
  tpi_ = pi_ + pi_;
  di_ = tpi_ / m_;
  dj_ = tpi_ / n_;
  pcf_ = pi_ * pi_ * a_ * a_ / (el_ * el_);
  fsdx_ = 4. / dx_;
  fsdy_ = 4. / dy_;
}

void SwCl12Benchmark::InitializeData() {
  InitPsiP();
  InitVelocities();

  // FIXME: Boundary conditions
  cl_int err;

  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;
  err = clEnqueueCopyBuffer(cmd_queue_, u_, u_curr_, 0, 0, sizeInBytes, 0, NULL,
                            NULL);
  err |= clEnqueueCopyBuffer(cmd_queue_, v_, v_curr_, 0, 0, sizeInBytes, 0,
                             NULL, NULL);
  err |= clEnqueueCopyBuffer(cmd_queue_, p_, p_curr_, 0, 0, sizeInBytes, 0,
                             NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueCopyBuffer");

  clFinish(cmd_queue_);
}

#define clFreeBuffer(buf)        \
  err = clReleaseMemObject(buf); \
  checkOpenCLErrors(err, "Failed to release cl buffer");
void SwCl12Benchmark::FreeBuffers() {
  cl_int err;
  clFreeBuffer(u_curr_);
  clFreeBuffer(u_next_);
  clFreeBuffer(v_curr_);
  clFreeBuffer(v_next_);
  clFreeBuffer(p_curr_);
  clFreeBuffer(p_next_);
  clFreeBuffer(u_);
  clFreeBuffer(v_);
  clFreeBuffer(p_);
  clFreeBuffer(cu_);
  clFreeBuffer(cv_);
  clFreeBuffer(z_);
  clFreeBuffer(h_);
  clFreeBuffer(psi_);
}
#undef clFreeBuffer

void SwCl12Benchmark::FreeKernels() {
  cl_int err;
  // Release kernels
  err = clReleaseKernel(kernel_sw_init_psi_p_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_init_psi_p_");
  err = clReleaseKernel(kernel_sw_init_velocities_);
  checkOpenCLErrors(err,
                    "Failed to release cl kernel: kernel_sw_init_velocities_");
  err = clReleaseKernel(kernel_sw_compute0_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_compute0_");
  err = clReleaseKernel(kernel_sw_update0_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_update0_");
  err = clReleaseKernel(kernel_sw_compute1_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_compute1_");
  err = clReleaseKernel(kernel_sw_update1_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_update1_");
  err = clReleaseKernel(kernel_sw_time_smooth_);
  checkOpenCLErrors(err, "Failed to release cl kernel: kernel_sw_time_smooth_");
}

void SwCl12Benchmark::InitPsiP() {
  cl_int err;

  err = clSetKernelArg(kernel_sw_init_psi_p_, 0, sizeof(double),
                       reinterpret_cast<void *>(&a_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 1, sizeof(double),
                        reinterpret_cast<void *>(&di_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 2, sizeof(double),
                        reinterpret_cast<void *>(&dj_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 3, sizeof(double),
                        reinterpret_cast<void *>(&pcf_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 4, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 5, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 6, sizeof(cl_mem),
                        reinterpret_cast<void *>(&p_));
  err |= clSetKernelArg(kernel_sw_init_psi_p_, 7, sizeof(cl_mem),
                        reinterpret_cast<void *>(&psi_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_init_psi_p_");

  const size_t globalSize[2] = {m_len_, n_len_};
  const size_t localSize[2] = {16, 16};

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_init_psi_p_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel                     "
                    "     kernel_sw_init_psi_p_");
}

void SwCl12Benchmark::InitVelocities() {
  cl_int err;

  const size_t globalSize[2] = {m_, n_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_init_velocities_, 0, sizeof(double),
                       reinterpret_cast<void *>(&dx_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 1, sizeof(double),
                        reinterpret_cast<void *>(&dy_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 3, sizeof(unsigned),
                        reinterpret_cast<void *>(&n_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 4, sizeof(cl_mem),
                        reinterpret_cast<void *>(&psi_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 5, sizeof(cl_mem),
                        reinterpret_cast<void *>(&u_));
  err |= clSetKernelArg(kernel_sw_init_velocities_, 6, sizeof(cl_mem),
                        reinterpret_cast<void *>(&v_));
  checkOpenCLErrors(err,
                    "Failed to set kernel args: kernel_sw_init_velocities_");

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_init_velocities_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_init_psi_p_");
}

void SwCl12Benchmark::Compute0() {
  cl_int err;

  const size_t globalSize[2] = {m_len_, n_len_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_compute0_, 0, sizeof(double),
                       reinterpret_cast<void *>(&fsdx_));
  err |= clSetKernelArg(kernel_sw_compute0_, 1, sizeof(double),
                        reinterpret_cast<void *>(&fsdy_));
  err |= clSetKernelArg(kernel_sw_compute0_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_compute0_, 3, sizeof(cl_mem),
                        reinterpret_cast<void *>(&u_));
  err |= clSetKernelArg(kernel_sw_compute0_, 4, sizeof(cl_mem),
                        reinterpret_cast<void *>(&v_));
  err |= clSetKernelArg(kernel_sw_compute0_, 5, sizeof(cl_mem),
                        reinterpret_cast<void *>(&p_));
  err |= clSetKernelArg(kernel_sw_compute0_, 6, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cu_));
  err |= clSetKernelArg(kernel_sw_compute0_, 7, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cv_));
  err |= clSetKernelArg(kernel_sw_compute0_, 8, sizeof(cl_mem),
                        reinterpret_cast<void *>(&z_));
  err |= clSetKernelArg(kernel_sw_compute0_, 9, sizeof(cl_mem),
                        reinterpret_cast<void *>(&h_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute0_");

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_compute0_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_compute0_");
}

void SwCl12Benchmark::PeriodicUpdate0() {
  cl_int err;

  const size_t globalSize[2] = {m_, n_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_update0_, 0, sizeof(unsigned),
                       reinterpret_cast<void *>(&m_));
  err |= clSetKernelArg(kernel_sw_update0_, 1, sizeof(unsigned),
                        reinterpret_cast<void *>(&n_));
  err |= clSetKernelArg(kernel_sw_update0_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_update0_, 3, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cu_));
  err |= clSetKernelArg(kernel_sw_update0_, 4, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cv_));
  err |= clSetKernelArg(kernel_sw_update0_, 5, sizeof(cl_mem),
                        reinterpret_cast<void *>(&z_));
  err |= clSetKernelArg(kernel_sw_update0_, 6, sizeof(cl_mem),
                        reinterpret_cast<void *>(&h_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update0_");

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_update0_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update0_");
}

void SwCl12Benchmark::Compute1() {
  tdts8_ = tdt_ / 8.;
  tdtsdx_ = tdt_ / dx_;
  tdtsdy_ = tdt_ / dy_;

  cl_int err;

  const size_t globalSize[2] = {m_len_, n_len_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_compute1_, 0, sizeof(double),
                       reinterpret_cast<void *>(&tdts8_));
  err |= clSetKernelArg(kernel_sw_compute1_, 1, sizeof(double),
                        reinterpret_cast<void *>(&tdtsdx_));
  err |= clSetKernelArg(kernel_sw_compute1_, 2, sizeof(double),
                        reinterpret_cast<void *>(&tdtsdy_));
  err |= clSetKernelArg(kernel_sw_compute1_, 3, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_compute1_, 4, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cu_));
  err |= clSetKernelArg(kernel_sw_compute1_, 5, sizeof(cl_mem),
                        reinterpret_cast<void *>(&cv_));
  err |= clSetKernelArg(kernel_sw_compute1_, 6, sizeof(cl_mem),
                        reinterpret_cast<void *>(&z_));
  err |= clSetKernelArg(kernel_sw_compute1_, 7, sizeof(cl_mem),
                        reinterpret_cast<void *>(&h_));
  err |= clSetKernelArg(kernel_sw_compute1_, 8, sizeof(cl_mem),
                        reinterpret_cast<void *>(&u_curr_));
  err |= clSetKernelArg(kernel_sw_compute1_, 9, sizeof(cl_mem),
                        reinterpret_cast<void *>(&v_curr_));
  err |= clSetKernelArg(kernel_sw_compute1_, 10, sizeof(cl_mem),
                        reinterpret_cast<void *>(&p_curr_));
  err |= clSetKernelArg(kernel_sw_compute1_, 11, sizeof(cl_mem),
                        reinterpret_cast<void *>(&u_next_));
  err |= clSetKernelArg(kernel_sw_compute1_, 12, sizeof(cl_mem),
                        reinterpret_cast<void *>(&v_next_));
  err |= clSetKernelArg(kernel_sw_compute1_, 13, sizeof(cl_mem),
                        reinterpret_cast<void *>(&p_next_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute1_");

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_compute1_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_compute1_");
}

void SwCl12Benchmark::PeriodicUpdate1() {
  cl_int err;

  const size_t globalSize[2] = {m_, n_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_update1_, 0, sizeof(unsigned),
                       reinterpret_cast<void *>(&m_));
  err |= clSetKernelArg(kernel_sw_update1_, 1, sizeof(unsigned),
                        reinterpret_cast<void *>(&n_));
  err |= clSetKernelArg(kernel_sw_update1_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArg(kernel_sw_update1_, 3, sizeof(cl_mem),
                        reinterpret_cast<void *>(&u_next_));
  err |= clSetKernelArg(kernel_sw_update1_, 4, sizeof(cl_mem),
                        reinterpret_cast<void *>(&v_next_));
  err |= clSetKernelArg(kernel_sw_update1_, 5, sizeof(cl_mem),
                        reinterpret_cast<void *>(&p_next_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update1_");

  err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_update1_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update1_");
}

void SwCl12Benchmark::TimeSmooth(int ncycle) {
  if (ncycle > 1) {
    cl_int err;

    const size_t globalSize[2] = {m_len_, n_len_};
    const size_t localSize[2] = {16, 16};

    err = clSetKernelArg(kernel_sw_time_smooth_, 0, sizeof(unsigned),
                         reinterpret_cast<void *>(&m_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 1, sizeof(unsigned),
                          reinterpret_cast<void *>(&n_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 2, sizeof(unsigned),
                          reinterpret_cast<void *>(&m_len_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 3, sizeof(double),
                          reinterpret_cast<void *>(&alpha_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 4, sizeof(cl_mem),
                          reinterpret_cast<void *>(&u_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 5, sizeof(cl_mem),
                          reinterpret_cast<void *>(&v_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 6, sizeof(cl_mem),
                          reinterpret_cast<void *>(&p_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 7, sizeof(cl_mem),
                          reinterpret_cast<void *>(&u_curr_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 8, sizeof(cl_mem),
                          reinterpret_cast<void *>(&v_curr_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 9, sizeof(cl_mem),
                          reinterpret_cast<void *>(&p_curr_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 10, sizeof(cl_mem),
                          reinterpret_cast<void *>(&u_next_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 11, sizeof(cl_mem),
                          reinterpret_cast<void *>(&v_next_));
    err |= clSetKernelArg(kernel_sw_time_smooth_, 12, sizeof(cl_mem),
                          reinterpret_cast<void *>(&p_next_));
    checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_time_smooth_");

    err = clEnqueueNDRangeKernel(cmd_queue_, kernel_sw_time_smooth_, 2, NULL,
                                 globalSize, localSize, 0, NULL, NULL);
    checkOpenCLErrors(
        err, "Failed to clEnqueueNDRangeKernel kernel_sw_time_smooth_");

  } else {
    cl_int err;

    tdt_ += tdt_;
    size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

    err = clEnqueueCopyBuffer(cmd_queue_, u_, u_curr_, 0, 0, sizeInBytes, 0,
                              NULL, NULL);
    err |= clEnqueueCopyBuffer(cmd_queue_, v_, v_curr_, 0, 0, sizeInBytes, 0,
                               NULL, NULL);
    err |= clEnqueueCopyBuffer(cmd_queue_, p_, p_curr_, 0, 0, sizeInBytes, 0,
                               NULL, NULL);
    err |= clEnqueueCopyBuffer(cmd_queue_, u_next_, u_, 0, 0, sizeInBytes, 0,
                               NULL, NULL);
    err |= clEnqueueCopyBuffer(cmd_queue_, v_next_, v_, 0, 0, sizeInBytes, 0,
                               NULL, NULL);
    err |= clEnqueueCopyBuffer(cmd_queue_, p_next_, p_, 0, 0, sizeInBytes, 0,
                               NULL, NULL);
    checkOpenCLErrors(err, "Failed to clEnqueueCopyBuffer");

    clFinish(cmd_queue_);
  }
}

void SwCl12Benchmark::Run() {
  for (unsigned i = 0; i < itmax_; ++i) {
    Compute0();
    PeriodicUpdate0();
    Compute1();
    PeriodicUpdate1();
    TimeSmooth(i);
  }
}

void SwCl12Benchmark::Cleanup() {
  SwBenchmark::Cleanup();
  cl_int err;

  // Release buffers
  FreeBuffers();

  // Release kernels
  FreeKernels();

  // Release program
  err = clReleaseProgram(program_);
  checkOpenCLErrors(err, "Release objects.\n");
}
