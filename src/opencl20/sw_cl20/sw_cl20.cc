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

#include "src/opencl20/sw_cl20/sw_cl20.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <memory>
#include <cmath>

ShallowWater::ShallowWater(unsigned m, unsigned n)
    : m_(m),
      n_(n),
      m_len_(m + 1),
      n_len_(n + 1),
      itmax_(250),
      dt_(90.),
      tdt_(dt_),
      dx_(100000.),
      dy_(100000.),
      a_(1000000.),
      alpha_(.001),
      el_(n_ * dx_),
      pi_(4. * atanf(1.)),
      tpi_(pi_ + pi_),
      di_(tpi_ / m_),
      dj_(tpi_ / n_),
      pcf_(pi_ * pi_ * a_ * a_ / (el_ * el_)),
      fsdx_(4. / dx_),
      fsdy_(4. / dy_) {
  runtime_ = clRuntime::getInstance();
  file_ = clFile::getInstance();

  platform_ = runtime_->getPlatformID();
  device_ = runtime_->getDevice();
  context_ = runtime_->getContext();
  cmdQueue_ = runtime_->getCmdQueue(0);
}

ShallowWater::~ShallowWater() {}

void ShallowWater::InitKernel() {
  cl_int err;

  // Open kernel file_
  file_->open("sw_Kernels.cl");

  // Create program
  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);

  // Create program with OpenCL 2.0 support
  err = clBuildProgram(program_, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0x10000, buf,
                          NULL);
    printf("\n%s\n", buf);
    exit(-1);
  }

  // Create kernels
  kernel_sw_init_psi_p_ = clCreateKernel(program_, "sw_init_psi_p", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_psi_p_");

  kernel_sw_init_velocities_ =
      clCreateKernel(program_, "sw_init_velocities", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_velocities_");

  kernel_sw_compute0_ = clCreateKernel(program_, "sw_compute0", &err);
  checkOpenCLErrors(err, "Failed to create sw_compute0");

  kernel_sw_update0_ = clCreateKernel(program_, "sw_update0", &err);
  checkOpenCLErrors(err, "Failed to create sw_periodic_update0");

  kernel_sw_compute1_ = clCreateKernel(program_, "sw_compute1", &err);
  checkOpenCLErrors(err, "Failed to create sw_compute1");

  kernel_sw_update1_ = clCreateKernel(program_, "sw_update1", &err);
  checkOpenCLErrors(err, "Failed to create sw_periodic_update1");

  kernel_sw_time_smooth_ = clCreateKernel(program_, "sw_time_smooth", &err);
  checkOpenCLErrors(err, "Failed to create sw_time_smooth");
}

void ShallowWater::InitBuffer() {
  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

  // Fine grain buffers
  u_curr_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  u_next_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  v_curr_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  v_next_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  p_curr_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  p_next_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  u_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  v_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  p_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  cu_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  cv_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  z_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  h_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
  psi_ = reinterpret_cast<double *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 sizeInBytes, 0));
}

void ShallowWater::FreeKernel() {
  cl_int err;

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

void ShallowWater::FreeBuffer() {
  clSVMFree(context_, u_curr_);
  clSVMFree(context_, u_next_);
  clSVMFree(context_, v_curr_);
  clSVMFree(context_, v_next_);
  clSVMFree(context_, p_curr_);
  clSVMFree(context_, p_next_);
  clSVMFree(context_, u_);
  clSVMFree(context_, v_);
  clSVMFree(context_, p_);
  clSVMFree(context_, cu_);
  clSVMFree(context_, cv_);
  clSVMFree(context_, z_);
  clSVMFree(context_, h_);
  clSVMFree(context_, psi_);
}

void ShallowWater::InitPsiP() {
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
  err |= clSetKernelArgSVMPointer(kernel_sw_init_psi_p_, 6,
                                  reinterpret_cast<void *>(p_));
  err |= clSetKernelArgSVMPointer(kernel_sw_init_psi_p_, 7,
                                  reinterpret_cast<void *>(psi_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_init_psi_p_");

  const size_t globalSize[2] = {m_len_, n_len_};
  const size_t localSize[2] = {16, 16};

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_init_psi_p_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_init_psi_p_");
}

void ShallowWater::InitVelocities() {
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
  err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities_, 4,
                                  reinterpret_cast<void *>(psi_));
  err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities_, 5,
                                  reinterpret_cast<void *>(u_));
  err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities_, 6,
                                  reinterpret_cast<void *>(v_));
  checkOpenCLErrors(err,
                    "Failed to set kernel args: kernel_sw_init_velocities_");

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_init_velocities_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_init_psi_p_");
}

void ShallowWater::Compute0() {
  cl_int err;

  const size_t globalSize[2] = {m_len_, n_len_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_compute0_, 0, sizeof(double),
                       reinterpret_cast<void *>(&fsdx_));
  err |= clSetKernelArg(kernel_sw_compute0_, 1, sizeof(double),
                        reinterpret_cast<void *>(&fsdy_));
  err |= clSetKernelArg(kernel_sw_compute0_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 3,
                                  reinterpret_cast<void *>(u_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 4,
                                  reinterpret_cast<void *>(v_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 5,
                                  reinterpret_cast<void *>(p_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 6,
                                  reinterpret_cast<void *>(cu_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 7,
                                  reinterpret_cast<void *>(cv_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 8,
                                  reinterpret_cast<void *>(z_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute0_, 9,
                                  reinterpret_cast<void *>(h_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute0_");

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_compute0_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_compute0_");
}

void ShallowWater::PeriodicUpdate0() {
  cl_int err;

  const size_t globalSize[2] = {m_, n_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_update0_, 0, sizeof(unsigned),
                       reinterpret_cast<void *>(&m_));
  err |= clSetKernelArg(kernel_sw_update0_, 1, sizeof(unsigned),
                        reinterpret_cast<void *>(&n_));
  err |= clSetKernelArg(kernel_sw_update0_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update0_, 3,
                                  reinterpret_cast<void *>(cu_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update0_, 4,
                                  reinterpret_cast<void *>(cv_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update0_, 5,
                                  reinterpret_cast<void *>(z_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update0_, 6,
                                  reinterpret_cast<void *>(h_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update0_");

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_update0_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update0_");
}

void ShallowWater::Compute1() {
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
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 4,
                                  reinterpret_cast<void *>(cu_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 5,
                                  reinterpret_cast<void *>(cv_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 6,
                                  reinterpret_cast<void *>(z_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 7,
                                  reinterpret_cast<void *>(h_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 8,
                                  reinterpret_cast<void *>(u_curr_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 9,
                                  reinterpret_cast<void *>(v_curr_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 10,
                                  reinterpret_cast<void *>(p_curr_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 11,
                                  reinterpret_cast<void *>(u_next_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 12,
                                  reinterpret_cast<void *>(v_next_));
  err |= clSetKernelArgSVMPointer(kernel_sw_compute1_, 13,
                                  reinterpret_cast<void *>(p_next_));

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_compute1_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err,
                    "Failed to clEnqueueNDRangeKernel kernel_sw_compute1_");
}

void ShallowWater::PeriodicUpdate1() {
  cl_int err;

  const size_t globalSize[2] = {m_, n_};
  const size_t localSize[2] = {16, 16};

  err = clSetKernelArg(kernel_sw_update1_, 0, sizeof(unsigned),
                       reinterpret_cast<void *>(&m_));
  err |= clSetKernelArg(kernel_sw_update1_, 1, sizeof(unsigned),
                        reinterpret_cast<void *>(&n_));
  err |= clSetKernelArg(kernel_sw_update1_, 2, sizeof(unsigned),
                        reinterpret_cast<void *>(&m_len_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update1_, 3,
                                  reinterpret_cast<void *>(u_next_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update1_, 4,
                                  reinterpret_cast<void *>(v_next_));
  err |= clSetKernelArgSVMPointer(kernel_sw_update1_, 5,
                                  reinterpret_cast<void *>(p_next_));
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update1_");

  err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_update1_, 2, NULL,
                               globalSize, localSize, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update1_");
}

void ShallowWater::TimeSmooth(int ncycle) {
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
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 4,
                                    reinterpret_cast<void *>(u_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 5,
                                    reinterpret_cast<void *>(v_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 6,
                                    reinterpret_cast<void *>(p_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 7,
                                    reinterpret_cast<void *>(u_curr_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 8,
                                    reinterpret_cast<void *>(v_curr_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 9,
                                    reinterpret_cast<void *>(p_curr_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 10,
                                    reinterpret_cast<void *>(u_next_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 11,
                                    reinterpret_cast<void *>(v_next_));
    err |= clSetKernelArgSVMPointer(kernel_sw_time_smooth_, 12,
                                    reinterpret_cast<void *>(p_next_));
    checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_time_smooth_");

    err = clEnqueueNDRangeKernel(cmdQueue_, kernel_sw_time_smooth_, 2, NULL,
                                 globalSize, localSize, 0, NULL, NULL);
    checkOpenCLErrors(
        err, "Failed to clEnqueueNDRangeKernel kernel_sw_time_smooth_");
  } else {
    cl_int err;

    tdt_ += tdt_;
    size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

    err = clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, u_curr_, u_, sizeInBytes, 0,
                             NULL, NULL);
    err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, v_curr_, v_, sizeInBytes, 0,
                              NULL, NULL);
    err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, p_curr_, p_, sizeInBytes, 0,
                              NULL, NULL);
    err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, u_, u_next_, sizeInBytes, 0,
                              NULL, NULL);
    err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, v_, v_next_, sizeInBytes, 0,
                              NULL, NULL);
    err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, p_, p_next_, sizeInBytes, 0,
                              NULL, NULL);
    checkOpenCLErrors(err, "Failed to clEnqueueSVMMemcpy");

    clFinish(cmdQueue_);
  }
}

void ShallowWater::Initialize() {
  InitKernel();
  InitBuffer();

  InitPsiP();
  InitVelocities();

  // FIXME: Boundary conditions
  cl_int err;

  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;
  err = clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, u_curr_, u_, sizeInBytes, 0,
                           NULL, NULL);
  err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, v_curr_, v_, sizeInBytes, 0,
                            NULL, NULL);
  err |= clEnqueueSVMMemcpy(cmdQueue_, CL_FALSE, p_curr_, p_, sizeInBytes, 0,
                            NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueSVMMemcpy");

  clFinish(cmdQueue_);
}

void ShallowWater::Run() {
  for (unsigned i = 0; i < itmax_; ++i) {
    Compute0();
    PeriodicUpdate0();
    Compute1();
    PeriodicUpdate1();
    TimeSmooth(i);
  }
}

void ShallowWater::Cleanup() {
  FreeBuffer();
  FreeKernel();
}
