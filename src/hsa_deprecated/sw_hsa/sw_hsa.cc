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

#include "src/hsa/sw_hsa/sw_hsa.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <memory>
#include <cmath>
#include <cstring>

#include "src/hsa/sw_hsa/kernels.h"

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
      fsdy_(4. / dy_) {}

ShallowWater::~ShallowWater() {}

void ShallowWater::InitKernel() {}

void ShallowWater::InitBuffer() {
  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

  // Fine grain buffers
  u_curr_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  u_next_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  v_curr_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  v_next_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  p_curr_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  p_next_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  u_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  v_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  p_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  cu_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  cv_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  z_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  h_ = reinterpret_cast<double *>(malloc(sizeInBytes));
  psi_ = reinterpret_cast<double *>(malloc(sizeInBytes));
}

void ShallowWater::FreeKernel() {}

void ShallowWater::FreeBuffer() {
  free(u_curr_);
  free(u_next_);
  free(v_curr_);
  free(v_next_);
  free(p_curr_);
  free(p_next_);
  free(u_);
  free(v_);
  free(p_);
  free(cu_);
  free(cv_);
  free(z_);
  free(h_);
  free(psi_);
}

void ShallowWater::InitPsiP() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_len_;
  lparm->gdims[1] = m_len_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_init_psi_p(a_, di_, dj_, pcf_, m_len_, m_len_, p_, psi_, lparm);
}

void ShallowWater::InitVelocities() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_;
  lparm->gdims[1] = m_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_init_velocities(dx_, dy_, m_, n_, psi_, u_, v_, lparm);
}

void ShallowWater::Compute0() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_len_;
  lparm->gdims[1] = m_len_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_compute0(fsdx_, fsdy_, m_len_, u_, v_, p_, cu_, cv_, z_, h_, lparm);
}

void ShallowWater::PeriodicUpdate0() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_;
  lparm->gdims[1] = n_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_update0(m_, n_, m_len_, cu_, cv_, z_, h_, lparm);
}

void ShallowWater::Compute1() {
  tdts8_ = tdt_ / 8.;
  tdtsdx_ = tdt_ / dx_;
  tdtsdy_ = tdt_ / dy_;

  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_len_;
  lparm->gdims[1] = m_len_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_compute1(tdts8_, tdtsdx_, tdtsdy_, m_len_, cu_, cv_, z_, h_, u_curr_,
              v_curr_, p_curr_, u_next_, v_next_, p_next_, lparm);
}

void ShallowWater::PeriodicUpdate1() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->gdims[0] = m_len_;
  lparm->gdims[1] = m_len_;
  lparm->ldims[0] = 16;
  lparm->ldims[1] = 16;

  sw_update1(m_, n_, m_len_, u_next_, v_next_, p_next_, lparm);
}

void ShallowWater::TimeSmooth(int ncycle) {
  if (ncycle > 1) {
    SNK_INIT_LPARM(lparm, 0);
    lparm->gdims[0] = m_len_;
    lparm->gdims[1] = m_len_;
    lparm->ldims[0] = 16;
    lparm->ldims[1] = 16;

    sw_time_smooth(m_, n_, m_len_, alpha_, u_, v_, p_, u_curr_, v_curr_,
                   p_curr_, u_next_, v_next_, p_next_, lparm);
  } else {
    tdt_ += tdt_;
    size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;

    memcpy(u_curr_, u_, sizeInBytes);
    memcpy(v_curr_, v_, sizeInBytes);
    memcpy(p_curr_, p_, sizeInBytes);
    memcpy(u_, u_next_, sizeInBytes);
    memcpy(v_, v_next_, sizeInBytes);
    memcpy(p_, p_next_, sizeInBytes);
  }
}

void ShallowWater::Initialize() {
  InitKernel();
  InitBuffer();

  InitPsiP();
  InitVelocities();

  size_t sizeInBytes = sizeof(double) * m_len_ * n_len_;
  memcpy(u_curr_, u_, sizeInBytes);
  memcpy(v_curr_, v_, sizeInBytes);
  memcpy(p_curr_, p_, sizeInBytes);
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

void ShallowWater::Cleanup() {}
