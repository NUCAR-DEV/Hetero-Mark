/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

float Phi(float X) {
  float y, absX, t;

  // the coefficients
  const float c1 = 0.319381530f;
  const float c2 = -0.356563782f;
  const float c3 = 1.781477937f;
  const float c4 = -1.821255978f;
  const float c5 = 1.330274429f;

  const float oneBySqrt2pi = 0.398942280f;

  absX = fabs(X);
  t = 1.0f / (1.0f + 0.2316419f * absX);

    y = 1.0f -
      oneBySqrt2pi * exp(-X * X / 2.0f) * t *
      (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));

    return (X < 0) ? (1.0f - y) : y;
}

__kernel void bs_cl12(__global float *rand_array, __global float *d_call_price_,
                        __global float *d_put_price_) {

  uint tid = get_global_id(0);
  
  // the variable representing the value in the array[i]
  float i_rand = rand_array[tid];

  // calculating the initial S,K,T, and R
  float s = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float k = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float t = 1.0 * i_rand + 10.0 * (1.0f - i_rand);
  float r = 0.01 * i_rand + 0.05 * (1.0f - i_rand);
  float sigma = 0.01 * i_rand + 0.10 * (1.0f - i_rand);

  // Calculating the sigmaSqrtT
  float sigma_sqrt_t_ = sigma * sqrt(t);

  // Calculating the derivatives
  float d1 = (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigma_sqrt_t_;
  float d2 = d1 - sigma_sqrt_t_;

  // Calculating exponent
  float k_exp_minus_rt_ = k * exp(-r * t);

  // Getting the output call and put prices
  d_call_price_[tid] = s * Phi(d1) - k_exp_minus_rt_ * Phi(d2);
  d_put_price_[tid] = k_exp_minus_rt_ * Phi(-d2) - s * Phi(-d1);
}
