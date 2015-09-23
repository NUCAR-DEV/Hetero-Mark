/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research
 * (NUCAR)
 * Group, Northeastern University,
 * http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy
 * of this software and associated documentation files (the
 * "Software"), to deal
 *  with the Software without restriction, including without
 * limitation
 * the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the
 * Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright
 *   notice, this
 *   list of conditions and the following disclaimers. Redistributions
 *   in binary
 *   form must reproduce the above copyright notice, this list of
 *   conditions and
 *   the following disclaimers in the documentation and/or other
 *   materials
 *   provided with the distribution. Neither the names of NUCAR,
 *   Northeastern
 *   University, nor the names of its contributors may be used to
 *   endorse or
 *   promote products derived from this Software without specific
 *   prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *   MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
 *   SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 *   DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *   ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *   OTHER
 *   DEALINGS WITH THE SOFTWARE.
 *
 * Shallow water Physics simulation engine
 *
 */

#include "include/sw.h"

#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <time.h>/* for clock_gettime */
#include <unistd.h>

#include <memory>
#include <cmath>

#define BILLION 1000000000L

void advance_spinner() {
  static char bars[] = { '/', '-', '\\', '|' };
  static int nbars = sizeof(bars) / sizeof(char);
  static int pos = 0;

  printf("%c\b", bars[pos]);
  fflush(stdout);
  pos = (pos + 1) % nbars;
}

ShallowWater::ShallowWater(unsigned _M, unsigned _N)
  :
  M(_M),
  N(_N),
  M_LEN(_M+1),
  N_LEN(_N+1),
  ITMAX(250),
  dt(90.),
  tdt(dt),
  dx(100000.),
  dy(100000.),
  fsdx(4. / dx),
  fsdy(4. / dy),
  a(1000000.),
  alpha(.001),
  el(N * dx),
  pi(4. * atanf(1.)),
  tpi(pi + pi),
  di(tpi / M),
  dj(tpi / N),
  pcf(pi * pi * a * a / (el * el)) {
  runtime  = clRuntime::getInstance();
  file     = clFile::getInstance();

  platform = runtime->getPlatformID();
  device   = runtime->getDevice();
  context  = runtime->getContext();
  cmdQueue = runtime->getCmdQueue(0);

  InitKernel();
  InitBuffer();
}

ShallowWater::~ShallowWater() {
  FreeKernel();
  FreeBuffer();
}

void ShallowWater::InitKernel() {
  cl_int err;

  // Open kernel file
  file->open("sw_Kernels.cl");

  // Create program
  const char *source = file->getSourceChar();
  program = clCreateProgramWithSource(context, 1,
                                      (const char **)&source, NULL, &err);

  // Create program with OpenCL 2.0 support
  err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0x10000,
                          buf,
                          NULL);
    printf("\n%s\n", buf);
    exit(-1);
  }

  // Create kernels
  kernel_sw_init_psi_p = clCreateKernel(program, "sw_init_psi_p", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_psi_p");

  kernel_sw_init_velocities = clCreateKernel(program,
                                             "sw_init_velocities", &err);
  checkOpenCLErrors(err, "Failed to create kernel_sw_init_velocities");

  kernel_sw_compute0 = clCreateKernel(program, "sw_compute0", &err);
  checkOpenCLErrors(err, "Failed to create sw_compute0");

  kernel_sw_update0 = clCreateKernel(program, "sw_update0", &err);
  checkOpenCLErrors(err, "Failed to create sw_periodic_update0");

  kernel_sw_compute1 = clCreateKernel(program, "sw_compute1", &err);
  checkOpenCLErrors(err, "Failed to create sw_compute1");

  kernel_sw_update1 = clCreateKernel(program, "sw_update1", &err);
  checkOpenCLErrors(err, "Failed to create sw_periodic_update1");

  kernel_sw_time_smooth = clCreateKernel(program, "sw_time_smooth", &err);
  checkOpenCLErrors(err, "Failed to create sw_time_smooth");
}

void ShallowWater::InitBuffer() {
  size_t sizeInBytes = sizeof(double) * M_LEN * N_LEN;

  cl_int err;
  u_curr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u_curr");
  u_next = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u_next");
  v_curr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v_curr");
  v_next = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v_next");
  p_curr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p_curr");
  p_next = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p_next");
  u      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: u");
  v      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: v");
  p      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: p");
  cu     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: cu");
  cv     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: cv");
  z      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: z");
  h      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: h");
  psi    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer: psi");
}

void ShallowWater::FreeKernel() {
}

#define clFreeBuffer(buf)                                       \
  err = clReleaseMemObject(buf);                                \
  checkOpenCLErrors(err, "Failed to release cl buffer");
void ShallowWater::FreeBuffer() {
  cl_int err;
  clFreeBuffer(u_curr);
  clFreeBuffer(u_next);
  clFreeBuffer(v_curr);
  clFreeBuffer(v_next);
  clFreeBuffer(p_curr);
  clFreeBuffer(p_next);
  clFreeBuffer(u);
  clFreeBuffer(v);
  clFreeBuffer(p);
  clFreeBuffer(cu);
  clFreeBuffer(cv);
  clFreeBuffer(z);
  clFreeBuffer(h);
  clFreeBuffer(psi);
}
#undef clFreeBuffer

void ShallowWater::Init() {
  InitPsiP();
  InitVelocities();

  // FIXME: Boundary conditions
  cl_int err;

  size_t sizeInBytes = sizeof(double) * M_LEN * N_LEN;
  err  = clEnqueueCopyBuffer(cmdQueue, u, u_curr, 0, 0,
                             sizeInBytes, 0, NULL, NULL);
  err |= clEnqueueCopyBuffer(cmdQueue, v, v_curr, 0, 0,
                             sizeInBytes, 0, NULL, NULL);
  err |= clEnqueueCopyBuffer(cmdQueue, p, p_curr, 0, 0,
                             sizeInBytes, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueCopyBuffer");

  clFinish(cmdQueue);
}

void ShallowWater::InitPsiP() {
  cl_int err;

  err  = clSetKernelArg(kernel_sw_init_psi_p, 0, sizeof(double), (void *)&a);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 1, sizeof(double), (void *)&di);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 2, sizeof(double), (void *)&dj);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 3, sizeof(double), (void *)&pcf);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 4,
                        sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 5,
                        sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 6, sizeof(cl_mem), (void *)&p);
  err |= clSetKernelArg(kernel_sw_init_psi_p, 7, sizeof(cl_mem), (void *)&psi);
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_init_psi_p");

  const size_t globalSize[2] = {M_LEN, N_LEN};
  const size_t localSize[2] = {16, 16};

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_init_psi_p,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel \
                          kernel_sw_init_psi_p");

  advance_spinner();
}

void ShallowWater::InitVelocities() {
  cl_int err;

  const size_t globalSize[2] = {M, N};
  const size_t localSize[2] = {16, 16};

  err  = clSetKernelArg(kernel_sw_init_velocities, 0,
                        sizeof(double), (void *)&dx);
  err |= clSetKernelArg(kernel_sw_init_velocities, 1,
                        sizeof(double), (void *)&dy);
  err |= clSetKernelArg(kernel_sw_init_velocities, 2,
                        sizeof(unsigned), (void *)&M);
  err |= clSetKernelArg(kernel_sw_init_velocities, 3,
                        sizeof(unsigned), (void *)&N);
  err |= clSetKernelArg(kernel_sw_init_velocities, 4,
                        sizeof(cl_mem), (void *)&psi);
  err |= clSetKernelArg(kernel_sw_init_velocities, 5,
                        sizeof(cl_mem), (void *)&u);
  err |= clSetKernelArg(kernel_sw_init_velocities, 6,
                        sizeof(cl_mem), (void *)&v);
  checkOpenCLErrors(err, "Failed to set kernel args: \
                          kernel_sw_init_velocities");

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_init_velocities,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel \
                          kernel_sw_init_psi_p");

  advance_spinner();
}

void ShallowWater::Compute0() {
  cl_int err;

  const size_t globalSize[2] = {M_LEN, N_LEN};
  const size_t localSize[2] = {16, 16};

  err  = clSetKernelArg(kernel_sw_compute0, 0, sizeof(double), (void *)&fsdx);
  err |= clSetKernelArg(kernel_sw_compute0, 1, sizeof(double), (void *)&fsdy);
  err |= clSetKernelArg(kernel_sw_compute0, 2,
                        sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_compute0, 3, sizeof(cl_mem), (void *)&u);
  err |= clSetKernelArg(kernel_sw_compute0, 4, sizeof(cl_mem), (void *)&v);
  err |= clSetKernelArg(kernel_sw_compute0, 5, sizeof(cl_mem), (void *)&p);
  err |= clSetKernelArg(kernel_sw_compute0, 6, sizeof(cl_mem), (void *)&cu);
  err |= clSetKernelArg(kernel_sw_compute0, 7, sizeof(cl_mem), (void *)&cv);
  err |= clSetKernelArg(kernel_sw_compute0, 8, sizeof(cl_mem), (void *)&z);
  err |= clSetKernelArg(kernel_sw_compute0, 9, sizeof(cl_mem), (void *)&h);
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute0");

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_compute0,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_compute0");

  advance_spinner();
}

void ShallowWater::PeriodicUpdate0() {
  cl_int err;

  const size_t globalSize[2] = {M, N};
  const size_t localSize[2] = {16, 16};

  err  = clSetKernelArg(kernel_sw_update0, 0, sizeof(unsigned), (void *)&M);
  err |= clSetKernelArg(kernel_sw_update0, 1, sizeof(unsigned), (void *)&N);
  err |= clSetKernelArg(kernel_sw_update0, 2, sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_update0, 3, sizeof(cl_mem), (void *)&cu);
  err |= clSetKernelArg(kernel_sw_update0, 4, sizeof(cl_mem), (void *)&cv);
  err |= clSetKernelArg(kernel_sw_update0, 5, sizeof(cl_mem), (void *)&z);
  err |= clSetKernelArg(kernel_sw_update0, 6, sizeof(cl_mem), (void *)&h);
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update0");

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_update0,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update0");

  advance_spinner();
}

void ShallowWater::Compute1() {
  tdts8 = tdt / 8.;
  tdtsdx = tdt / dx;
  tdtsdy = tdt / dy;

  cl_int err;

  const size_t globalSize[2] = {M_LEN, N_LEN};
  const size_t localSize[2] = {16, 16};

  err  = clSetKernelArg(kernel_sw_compute1, 0, sizeof(double), (void *)&tdts8);
  err |= clSetKernelArg(kernel_sw_compute1, 1, sizeof(double), (void *)&tdtsdx);
  err |= clSetKernelArg(kernel_sw_compute1, 2, sizeof(double), (void *)&tdtsdy);
  err |= clSetKernelArg(kernel_sw_compute1, 3,
                        sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_compute1, 4, sizeof(cl_mem), (void *)&cu);
  err |= clSetKernelArg(kernel_sw_compute1, 5, sizeof(cl_mem), (void *)&cv);
  err |= clSetKernelArg(kernel_sw_compute1, 6, sizeof(cl_mem), (void *)&z);
  err |= clSetKernelArg(kernel_sw_compute1, 7, sizeof(cl_mem), (void *)&h);
  err |= clSetKernelArg(kernel_sw_compute1, 8, sizeof(cl_mem), (void *)&u_curr);
  err |= clSetKernelArg(kernel_sw_compute1, 9, sizeof(cl_mem), (void *)&v_curr);
  err |= clSetKernelArg(kernel_sw_compute1, 10,
                        sizeof(cl_mem), (void *)&p_curr);
  err |= clSetKernelArg(kernel_sw_compute1, 11,
                        sizeof(cl_mem), (void *)&u_next);
  err |= clSetKernelArg(kernel_sw_compute1, 12,
                        sizeof(cl_mem), (void *)&v_next);
  err |= clSetKernelArg(kernel_sw_compute1, 13,
                        sizeof(cl_mem), (void *)&p_next);
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute1");

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_compute1,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_compute1");

  advance_spinner();
}

void ShallowWater::PeriodicUpdate1() {
  cl_int err;

  const size_t globalSize[2] = {M, N};
  const size_t localSize[2] = {16, 16};

  err  = clSetKernelArg(kernel_sw_update1, 0, sizeof(unsigned), (void *)&M);
  err |= clSetKernelArg(kernel_sw_update1, 1, sizeof(unsigned), (void *)&N);
  err |= clSetKernelArg(kernel_sw_update1, 2, sizeof(unsigned), (void *)&M_LEN);
  err |= clSetKernelArg(kernel_sw_update1, 3, sizeof(cl_mem), (void *)&u_next);
  err |= clSetKernelArg(kernel_sw_update1, 4, sizeof(cl_mem), (void *)&v_next);
  err |= clSetKernelArg(kernel_sw_update1, 5, sizeof(cl_mem), (void *)&p_next);
  checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_update1");

  err = clEnqueueNDRangeKernel(cmdQueue,
                               kernel_sw_update1,
                               2,
                               NULL,
                               globalSize,
                               localSize,
                               0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_update1");

  advance_spinner();
}

void ShallowWater::TimeSmooth(int ncycle) {
  if (ncycle > 1) {
    cl_int err;

    const size_t globalSize[2] = {M_LEN, N_LEN};
    const size_t localSize[2] = {16, 16};

    err  = clSetKernelArg(kernel_sw_time_smooth, 0,
                          sizeof(unsigned), (void *)&M);
    err |= clSetKernelArg(kernel_sw_time_smooth, 1,
                          sizeof(unsigned), (void *)&N);
    err |= clSetKernelArg(kernel_sw_time_smooth, 2,
                          sizeof(unsigned), (void *)&M_LEN);
    err |= clSetKernelArg(kernel_sw_time_smooth, 3,
                          sizeof(double), (void *)&alpha);
    err |= clSetKernelArg(kernel_sw_time_smooth, 4, sizeof(cl_mem), (void *)&u);
    err |= clSetKernelArg(kernel_sw_time_smooth, 5, sizeof(cl_mem), (void *)&v);
    err |= clSetKernelArg(kernel_sw_time_smooth, 6, sizeof(cl_mem), (void *)&p);
    err |= clSetKernelArg(kernel_sw_time_smooth, 7,
                          sizeof(cl_mem), (void *)&u_curr);
    err |= clSetKernelArg(kernel_sw_time_smooth, 8,
                          sizeof(cl_mem), (void *)&v_curr);
    err |= clSetKernelArg(kernel_sw_time_smooth, 9,
                          sizeof(cl_mem), (void *)&p_curr);
    err |= clSetKernelArg(kernel_sw_time_smooth, 10,
                          sizeof(cl_mem), (void *)&u_next);
    err |= clSetKernelArg(kernel_sw_time_smooth, 11,
                          sizeof(cl_mem), (void *)&v_next);
    err |= clSetKernelArg(kernel_sw_time_smooth, 12,
                          sizeof(cl_mem), (void *)&p_next);
    checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_time_smooth");

    err = clEnqueueNDRangeKernel(cmdQueue,
                                 kernel_sw_time_smooth,
                                 2,
                                 NULL,
                                 globalSize,
                                 localSize,
                                 0, NULL, NULL);
    checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel \
                            kernel_sw_time_smooth");

    advance_spinner();
  } else {
    cl_int err;

    tdt += tdt;
    size_t sizeInBytes = sizeof(double) * M_LEN * N_LEN;

    err  = clEnqueueCopyBuffer(cmdQueue, u, u_curr, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    err |= clEnqueueCopyBuffer(cmdQueue, v, v_curr, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    err |= clEnqueueCopyBuffer(cmdQueue, p, p_curr, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    err |= clEnqueueCopyBuffer(cmdQueue, u_next, u, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    err |= clEnqueueCopyBuffer(cmdQueue, v_next, v, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    err |= clEnqueueCopyBuffer(cmdQueue, p_next, p, 0, 0,
                               sizeInBytes, 0, NULL, NULL);
    checkOpenCLErrors(err, "Failed to clEnqueueCopyBuffer");

    clFinish(cmdQueue);
  }
}

void ShallowWater::Run() {
  std::cout << "Running... ";

  Init();

  for (int i = 0; i < ITMAX; ++i) {
    Compute0();
    PeriodicUpdate0();
    Compute1();
    PeriodicUpdate1();
    TimeSmooth(i);
  }

  std::cout << std::endl;
}
