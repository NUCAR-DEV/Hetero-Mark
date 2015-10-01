// Kernel parameter order :
//							const parameters,
//							__global const data,
//							__global output data

//-----------------------------------------------------------------------------------------------//
//                                       Forward kernels
//-----------------------------------------------------------------------------------------------//
__kernel void FWD_init_alpha(const int    N,
                             __global float *b,
                             __global float *prior,
                             __global float *alpha,
                             __global float *beta) {
    size_t idx = get_global_id(0);
    if (idx < N) {
        alpha[idx] = prior[idx] * b[idx];
        beta[idx] = 1.0f; // for backward
    }
}



__kernel void  FWD_norm_alpha(const int N,
                              const int startpos,
                              __local  float *sm,
                              __global float *alpha,
                              __global float *lll)

{
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t bls = get_local_size(0);
    size_t gls = get_global_size(0);

    float tidsum = 0.f;
    int i;
    for(i=gid; i<N; i+=gls) {
        tidsum += alpha[startpos + i];
    }

    sm[tid] = tidsum;

    barrier(CLK_LOCAL_MEM_FENCE);


    // sum the value from each thread using shared memory
    if(bls >= 512) {
        if(tid < 256) {
            sm[tid] += sm[tid + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 256) {
        if(tid < 128) {
            sm[tid] += sm[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 128) {
        if(tid <  64) {
            sm[tid] += sm[tid +  64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  64) {
        if(tid <  32) {
            sm[tid] += sm[tid +  32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  32) {
        if(tid <  16) {
            sm[tid] += sm[tid +  16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  16) {
        if(tid <   8) {
            sm[tid] += sm[tid +   8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   8) {
        if(tid <   4) {
            sm[tid] += sm[tid +   4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   4) {
        if(tid <   2) {
            sm[tid] += sm[tid +   2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   2) {
        if(tid <   1) {
            sm[tid] += sm[tid +   1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    // element-wise division
    for(i=gid; i<N; i+=gls) {
        alpha[startpos + i] /= sm[0];
    }

    if(gid == 0) {
        lll[0] += log(sm[0]);
    }
}

__kernel void TransposeSym (const int N,
                            __local float *sm,
                            __global float *a,
                            __global float *aT) {
    // x : col			y: row
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    //size_t gx = get_group_id(0) * 16 + lx;
    //size_t gy = get_group_id(1) * 16 + ly;
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    //  width, height
    if((gx < N)	&& (gy < N)) {
        size_t index_in = gy * N + gx;
        sm[ly * 17 + lx] = a[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // transposed block index
    gx = get_group_id(1) * 16 + lx;
    gy = get_group_id(0) * 16 + ly;

    // height, width
    if((gx < N)	&& (gy < N)) {
        size_t index_out = gy * N + gx;
        aT[index_out] = sm[lx * 17 + ly];
    }
}



// TODO: use OpenCL 2.0 workgroup function instead
__kernel void FWD_update_alpha(const int N,
                               const int current,
                               __local float *sm,
                               __constant float *constMem,
                               __global float *aT,
                               __global float *b,
                               __global float *alpha) {
    // col
    size_t lx = get_local_id(0);
    size_t gx = get_global_id(0);

    // row
    size_t ly = get_local_id(1);
    size_t gy = get_global_id(1);

    // to iterate through columns
    int iters = N / 16;

    float data = 0.f;

    int i;
    for(i=0; i<iters; ++i) {
        int col = i * 16 + lx;
        data += aT[gy * N + col] * constMem[col];
    }

    sm[ly * 17 + lx] = data;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gx == 0) { // only first column exectues
        int start = ly * 17;

        data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
                + sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
                + sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
                + sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];


        alpha[current + gy] = data * b[current + gy];
    }
}


//-----------------------------------------------------------------------------------------------//
//                                         Backward kernels
//-----------------------------------------------------------------------------------------------//
__kernel void BK_BetaB(const int N,
                       const int pos,
                       __global const float *beta,
                       __global const float *b,
                       __global float *betaB) {
    size_t idx = get_global_id(0);
    if (idx < N) {
        betaB[idx] = b[pos + idx] * beta[pos + idx];
    }
}


__kernel void BK_update_beta(const int N,
                             const int current,
                             __local float *sm,
                             __constant float *constMem,
                             __global float *a,
                             __global float *beta) {
    // col
    size_t lx = get_local_id(0);
    size_t gx = get_global_id(0);

    // row
    size_t ly = get_local_id(1);
    size_t gy = get_global_id(1);

    // to iterate through columns
    int iters = N / 16;

    float data = 0.f;

    int i;
    for(i=0; i<iters; ++i) {
        int col = i * 16 + lx;
        data += a[gy * N + col] * constMem[col];
    }

    sm[ly * 17 + lx] = data;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gx == 0) { // only first column exectues
        int start = ly * 17;

        data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
                + sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
                + sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
                + sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];


        beta[current + gy] = data;
    }
}


__kernel void BK_norm_beta(const int N,
                           const int current,
                           __local  float *sm,
                           __global float *beta)

{
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t bls = get_local_size(0);
    size_t gls = get_global_size(0);

    float tidsum = 0.f;
    int i;
    for(i=gid; i<N; i+=gls) {
        tidsum += beta[current + i];
    }

    sm[tid] = tidsum;

    barrier(CLK_LOCAL_MEM_FENCE);


    // sum the value from each thread using shared memory
    if(bls >= 512) {
        if(tid < 256) {
            sm[tid] += sm[tid + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 256) {
        if(tid < 128) {
            sm[tid] += sm[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 128) {
        if(tid <  64) {
            sm[tid] += sm[tid +  64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  64) {
        if(tid <  32) {
            sm[tid] += sm[tid +  32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  32) {
        if(tid <  16) {
            sm[tid] += sm[tid +  16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  16) {
        if(tid <   8) {
            sm[tid] += sm[tid +   8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   8) {
        if(tid <   4) {
            sm[tid] += sm[tid +   4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   4) {
        if(tid <   2) {
            sm[tid] += sm[tid +   2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   2) {
        if(tid <   1) {
            sm[tid] += sm[tid +   1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    // element-wise division
    for(i=gid; i<N; i+=gls) {
        beta[current + i] /= sm[0];
    }
}


//-----------------------------------------------------------------------------------------------//
//                                       EM kernels
//-----------------------------------------------------------------------------------------------//

// Compute beta * B and alpha * beta
__kernel void EM_betaB_alphabeta(const int N,
                                 const int current,
                                 const int previous,
                                 __global const float *beta,
                                 __global const float *b,
                                 __global const float *alpha,
                                 __global       float *betaB,
                                 __global       float *alpha_beta) {
    size_t idx = get_global_id(0);
    if (idx < N) {
        betaB[idx]     = beta[previous + idx] * b[previous + idx];
        alpha_beta[idx] = beta[current + idx] * alpha[current + idx];
    }
}

__kernel void EM_update_gamma(const int N,
                              const int current,
                              __local float *sm,
                              __global const float *alpha_beta,
                              __global       float *gamma) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t bls = get_local_size(0);
    size_t gls = get_global_size(0);

    float tidsum = 0.f;
    int i;
    for(i=gid; i<N; i+=gls) {
        tidsum += alpha_beta[i];
    }

    sm[tid] = tidsum;

    barrier(CLK_LOCAL_MEM_FENCE);


    // sum the value from each thread using shared memory
    if(bls >= 512) {
        if(tid < 256) {
            sm[tid] += sm[tid + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 256) {
        if(tid < 128) {
            sm[tid] += sm[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 128) {
        if(tid <  64) {
            sm[tid] += sm[tid +  64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  64) {
        if(tid <  32) {
            sm[tid] += sm[tid +  32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  32) {
        if(tid <  16) {
            sm[tid] += sm[tid +  16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  16) {
        if(tid <   8) {
            sm[tid] += sm[tid +   8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   8) {
        if(tid <   4) {
            sm[tid] += sm[tid +   4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   4) {
        if(tid <   2) {
            sm[tid] += sm[tid +   2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   2) {
        if(tid <   1) {
            sm[tid] += sm[tid +   1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // element-wise division
    for(i=gid; i<N; i+=gls) {
        gamma[current + i] /= sm[0];
    }

}

// Compute alpha * betaB'
__kernel void EM_alpha_betaB(const int N,
                             const int current,
                             __global const float *betaB,
                             __global const float *alpha,
                             __global       float *alpha_betaB) {
    size_t gx = get_global_id(0); // col
    size_t gy = get_global_id(1); // row

    if(gx < N && gy < N) {
        alpha_betaB[gy * N + gx] = alpha[current + gy] * betaB[gx];
    }
}


// Compute A .* alpha_betaB
__kernel void EM_pre_xisum(const int N,
                           __local        float *sm,
                           __global const float *a,
                           __global const float *alpha_betaB,
                           __global float *xi_sum_tmp,
                           __global float *blk_result) {
    // local
    size_t lx = get_local_id(0);  // col
    size_t ly = get_local_id(1);  // row

    // global
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    float data = 0.f;

    if(gx < N && gy < N) {
        data = xi_sum_tmp[gy * N + gx] = a[gy * N + gx] * alpha_betaB[gy * N + gx];
    }

    sm[ly * 17 + lx] = data;

    barrier(CLK_LOCAL_MEM_FENCE);

    // summarize the data in local memory
    // assume block is 16 x 16
    size_t index = ly * 17 + lx;
    if(lx < 8) {
        sm[index] += sm[index + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lx < 4) {
        sm[index] += sm[index + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lx < 2) {
        sm[index] += sm[index + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lx < 1) {
        sm[index] += sm[index + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // TODO: atomic
    if(lx == 0 && ly == 0) {
        int index_out = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        blk_result[index_out] = sm[0] + sm[17] + sm[34] + sm[51]
                                + sm[68] + sm[85] + sm[102] + sm[119]
                                + sm[136] + sm[153] + sm[170] + sm[187]
                                + sm[204] + sm[221] + sm[238] + sm[255];
    }

}

__kernel void EM_update_xisum(const int N,
                              const float sum,
                              __global const float *xi_sum_tmp,
                              __global       float *xi_sum) {
    size_t gx = get_global_id(0); // col
    size_t gy = get_global_id(1); // row

    if(gx < N && gy < N) {
        size_t index = gy * N + gx;
        xi_sum[index] += xi_sum_tmp[index] / sum;
    }
}


// TODO
__kernel void EM_gamma(const int N,
                       const int current,
                       __local float *sm,
                       __global const float *alpha,
                       __global const float *beta,
                       __global       float *gamma) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t bls = get_local_size(0);
    size_t gls = get_global_size(0);

    float tidsum = 0.f;
    int i;
    for(i=gid; i<N; i+=gls) {
        tidsum += alpha[current + i] * beta[current + i];
    }

    sm[tid] = tidsum;

    barrier(CLK_LOCAL_MEM_FENCE);


    // sum the value from each thread using shared memory
    if(bls >= 512) {
        if(tid < 256) {
            sm[tid] += sm[tid + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 256) {
        if(tid < 128) {
            sm[tid] += sm[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >= 128) {
        if(tid <  64) {
            sm[tid] += sm[tid +  64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  64) {
        if(tid <  32) {
            sm[tid] += sm[tid +  32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  32) {
        if(tid <  16) {
            sm[tid] += sm[tid +  16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=  16) {
        if(tid <   8) {
            sm[tid] += sm[tid +   8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   8) {
        if(tid <   4) {
            sm[tid] += sm[tid +   4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   4) {
        if(tid <   2) {
            sm[tid] += sm[tid +   2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(bls >=   2) {
        if(tid <   1) {
            sm[tid] += sm[tid +   1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // element-wise division
    for(i=gid; i<N; i+=gls) {
        gamma[current + i] /= sm[0];
    }

}


__kernel void EM_expectA(const int N,
                         __local float *sm,
                         __global const float *xi_sum,
                         __global       float *expect_A) {
    size_t gx = get_global_id(0);
    size_t lx = get_local_id(0); // col

    size_t gy = get_global_id(1);
    size_t ly = get_local_id(1); // row

    float data = 0.f;

    int start = ly * 17;
    size_t offset = gy * N;

    int i;
    for(i = gx; i < N; i += 16) {
        data += xi_sum[offset + i];
    }

    sm[start + lx] = data;

    barrier(CLK_LOCAL_MEM_FENCE);

    // sum across rows
    if(gx == 0) { // only 16 threads are alive now
        data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
                + sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
                + sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
                + sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];

        if (data == 0.f) data = 1.f;

        // same the row sum at the 1st col of sm
        sm[start] = data;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(i = gx; i < N; i += 16) {
        expect_A[offset + i] = xi_sum[offset + i] / sm[start];
    }

}


__kernel void EM_gamma_state_sum(const int N,
                                 const int T,
                                 __local float *sm,
                                 __global const float *gamma,
                                 __global       float *gamma_state_sum) {
    // gamma :   T x N

    size_t gx = get_global_id(0); // col
    size_t gy = get_global_id(1); // row

    size_t lx = get_local_id(0); // col
    size_t ly = get_local_id(1); // row

    float data = 0.f;
    // sum along column
    int i;
    for(i = gy; i < T; i += 16) {
        data += gamma[i * N + gx];
    }

    // transpose in sm
    sm[lx * 17 + ly] = data;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gy == 0) { // only 16 threads are alive now
        int start = lx * 17;
        data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
                + sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
                + sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
                + sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];

        if (data == 0.f) data = 1.f;

        gamma_state_sum[gx] = data;
    }
}


__kernel void EM_gamma_obs(const int D,
                           const int T,
                           __constant float *constMem,
                           __global const float *observations, // D x T
                           __global       float *gamma_obs) {
    size_t gx = get_global_id(0); // col: T
    size_t gy = get_global_id(1); // row: D

    size_t id = gy * T + gx;

    if(gx < T && gy < D)
        gamma_obs[id] = observations[id] * constMem[gx];
}



__kernel void EM_expect_mu(const int D,
                           const int T,
                           const int offset,
                           const int hs,
                           __local float *sm,
                           __global const float *gamma_obs, // D x T
                           __global const float *gamma_state_sum,
                           __global       float *expect_mu) { // N x D
    // row-wise sum on gamma_obs (D x T)

    size_t  gx = get_global_id(0); // col
    size_t  gy = get_global_id(1); // row

    size_t  lx = get_local_id(0); // col
    size_t  ly = get_local_id(1); // row

    size_t stride = gy * T;

    float data = 0.f;

    int i;
    for(i = gx; i < T; i+=16) {
        data += gamma_obs[stride + i];
    }

    sm[ly * 17 + lx]= data;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gx == 0) { // only 16 threads are alive now
        int start = ly * 17;
        data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
                + sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
                + sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
                + sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];

        if(gy < D)
            expect_mu[offset + gy] = data / gamma_state_sum[hs];
    }

}


__kernel void EM_sigma_dev(const int D,
                           const int T,
                           const int hs,
                           __constant float *constMem,
                           __global const float *gamma_obs,
                           __global const float *observations,
                           __global const float *gamma_state_sum,
                           __global       float *sigma_dev) {
    // C = A x B'
    // C , sigma_dev  DxD
    // A , gamma_obs  DxT
    // B , observations DxT

    __local float lds_a[72]; // gamma_obs
    __local float lds_b[72]; // observations

    int lx = get_local_id(0); // col
    int ly = get_local_id(1); // row

    size_t gx = get_global_id(0); // col
    size_t gy = get_global_id(1); // row

    float sum = 0.f; // output sum for (gy, gx)

    int iter = T/8; // T is multiple of 8
    int m;
    for(m=0; m<iter; ++m) {
        // each iteration, load 8 x 8
        lds_a[ly * 9 + lx] = gamma_obs[gy * T + (lx + m * 8)];
        lds_b[ly * 9 + lx] = observations[gx * T + (ly + m * 8)];

        barrier(CLK_LOCAL_MEM_FENCE);

        // matrix mul on LDS
        // a x b
        int kk;
#pragma unroll
        for (kk = 0; kk < 8; ++kk) {
            sum += lds_a[ly * 9 + kk] * lds_b[lx * 9 + kk];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // sum is the mm result of gamma_obs * obs_t
    // sum * gamma_state_sum(s) - expect_mu(s) * expect_mu(s)'
    sigma_dev[gy * D + gx] = sum / gamma_state_sum[hs] - constMem[gy] * constMem[gx];
}


__kernel void EM_expect_sigma(const int blk_rows,
                              const int width,
                              const size_t start,
                              __global const float *sigma_dev,
                              __global       float *expect_sigma) {

    volatile __local int2 blkid;
    // find the subdiagnoal blocks
    // for example,
    // 0
    // 1        2
    // 3        4        5
    // 6        7        8        9

    // (0.0)
    // (1,0)        (1,1)
    // (2,0)        (2,1)        (2,2)
    // (3,0)        (3,1)        (3,2)        (3,3)

    // here , each block is [TILE][TILE]
    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);

    int bn = get_group_id(1);


    if(lid_x == 0 && lid_y == 0) {
        int i;
        int upper,lower;

        for(i = 2, lower = 0, upper = 1 ; i <= (blk_rows+1) ; i++) {
            if( bn >= lower && bn < upper) {
                blkid.y = i-2; // rows
                blkid.x = bn - lower; // cols
                break;
            } else {
                lower = upper;
                upper = upper + i;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // find the corresponding global thread index
    size_t gx, gy, gid;
    gx = blkid.x * 16 + lid_x;  // global column index
    gy = blkid.y * 16 + lid_y;
    gid = gy * width + gx;


    size_t gid_sym = gx * width + gy;

    float a = sigma_dev[gid];
    float b = sigma_dev[gid_sym];

    if(gx == gy) {
        a = a + 0.01f;
        b = b + 0.01f;
    }

    if( a > b ) {
        expect_sigma[start + gid] = a;
        expect_sigma[start + gid_sym] = a;
    } else {
        expect_sigma[start + gid] = b;
        expect_sigma[start + gid_sym] = b;
    }

}
