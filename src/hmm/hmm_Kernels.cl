// Forward kernels
__kernel void FWD_init_alpha(__global const float *b_d,
                             __global const float *pi_d,
                                      const int    N,
                             __global       float *alpha_d,
                             __global       float *ones_d,
                             __global       float *beta_d)
{
        unsigned int idx = get_global_id(0);
        if (idx < N) {
                alpha_d[idx] = pi_d[idx] * b_d[idx];
                beta_d[idx] = ones_d[idx] = 1.0f; // for backward
        }
}


__kernel void FWD_scaling(         const int    N,
                          __global       float *alpha_d
                          __global const float *scale_factor,
                                   const int t,
                          )
{
        unsigned int idx = get_global_id(0);

        if (idx < N) {
                alpha_d[idx] /= scale_factor[t];
        }
}


__kernel void FWD_calc_alpha(         const int N,
                             __global const float *b_d,
                             __global float *alpha_d) 
{
        unsigned int idx = get_global_id(0);

        if (idx < N) {
                alpha_d[idx] *= b_d[idx];
        }
}

// TODO: use OpenCL 2.0 workgroup function instead
__kernel void FWD_sum_ll(         const int T,
                         __global float *ll_d)
{
        uint lid = get_local_id(0);
        uint gid = get_global_id(0);

        // T = 64
        __local float sm[64];

        if (gid < T){
                sm[lid] = log10(ll_d[gid]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        //reduction
        if (lid < 32) {
                __local float *smem = sm;
                smem[lid] += smem[lid + 32];
                smem[lid] += smem[lid + 16];
                smem[lid] += smem[lid +  8];
                smem[lid] += smem[lid +  4];
                smem[lid] += smem[lid +  2];
                smem[lid] += smem[lid +  1];
        }

        if (lid == 0) {
                ll_d[T] = sm[0];
        }
}

// Backward kernels
__kernel void BK_update_beta(__global const float *beta_d,
                             __global const float *B_d,
                                      const int N,
                             __global float *betaB_d)
{
        unsigned int idx = get_global_id(0);
        if (idx < N) {
                betaB_d[idx] = B_d[idx] * beta_d[idx];
        }
}


__kernel void BK_scaling(         const int N,
                         __global const float *ll_d,
                         __global float *beta)
{
        unsigned int idx = get_global_id(0);

        if (idx < N) {
                beta[idx] /= ll_d[0];
        }
}

// BW kernels
#ifndef TILE
        #define TILE 16 // 2D Kernel Tiling
#endif

#ifndef SIZE 
        #define SIZE 4096 
#endif

// cache global memory
// __constant float ConstA[SIZE]; 
// __constant float ConstB[SIZE]; 

// __constant float gamma_state_sumC[SIZE];
// __constant float bufferT[64];
// __constant float expect_mu_state[64];


__kernel void EM_betaB_alphabeta(__global const float *beta, 
                                 __global const float *B, 
                                 __global       float *betaB,  
                                 __global const float *alpha,
                                 __global       float *alpha_beta,
                                          const int N,
                                          const int current, 
                                          const int previous)
{
        uint idx = get_global_id(0);
        if (idx < N) {
                betaB[idx]     = beta[previous + idx] * B[previous + idx];
                alpha_beta[idx] = beta[current + idx] * alpha[current + idx];
        }
}


__kernel void EM_alphabeta_update_gamma(__global const float *alpha_beta, 
                                        __global       float *gamma,
                                        __global const float *ll_d, 
                                                 const int N,
                                                 const uint current)
{
        uint idx = get_global_id(0);
        if (idx < N){
                gamma[current + idx] = alpha_beta[idx] / ll_d[0];
        }
}

__kernel void EM_A_mul_alphabetaB(__global const float *A, 
                                  __global       float *A_alphabetaB,
                                  __global       float *blk_result,
                                  __constant     float *ConstA,
                                  __constant     float *ConstB,
                                           const int N) 
{

        uint lx = get_local_id(0); // col  
        uint ly = get_local_id(1); // row 

        uint gx = get_group_id(0) * get_local_size(0) + lx;
        uint gy = get_group_id(1) * get_local_size(1) + ly;

        float data;

        uint outID = gy * N + gx;
        volatile __local float lds[256];

        // localsize: 16 x 16
        // alphabetaB[i][j] = alpha[i] * betaB[j];
        // A[i][j] .* alphabetaB[i][j];

        // data = A_alphabetaB[gy * N + gx] = A[gy * N + gx] * alpha[current + gy] * betaB[gx];
        data = A[outID] * ConstA[gy] * ConstB[gx];
        A_alphabetaB[outID] = data;

        // lds[ly][lx]
        uint index = ly * TILE + lx;
        lds[index] = data;

        barrier(CLK_LOCAL_MEM_FENCE);

        //reduction
        if(lx < 8) {lds[index] += lds[index + 8];}
        if(lx < 4) {lds[index] += lds[index + 4];}
        if(lx < 2) {lds[index] += lds[index + 2];}
        if(lx < 1) {lds[index] += lds[index + 1];}
        if(lx == 0 && ly == 0){
                int id = get_group_id(1) * get_local_size(0) + get_group_id(0);
                blk_result[id] = lds[0] + lds[16] + lds[32] + lds[48] 
                        + lds[64] + lds[80] + lds[96] + lds[112]
                        + lds[128] + lds[144] + lds[160] + lds[176] 
                        + lds[192] + lds[208] + lds[224] + lds[240];
        }
}


__kernel void EM_update_xisum(__global const float *A_alphabetaB,
                              __global       float *xi_sum,
                                       const float sum,
                                       const int N) 
{
        uint gx = get_global_id(0);
        uint gy = get_global_id(1);
        uint outID = gy * N + gx;
        xi_sum[outID] += A_alphabetaB[outID] / sum;
}

__kernel void EM_alphabeta(__global const float *beta, 
                           __global const float *alpha,
                           __global       float *alpha_beta,
                                    const int N)
{
        uint idx = get_global_id(0);
        if (idx < N) {
                alpha_beta[idx] = beta[idx] * alpha[idx];
        }
}

// expected_A     = mk_stochastic(xi_sum);
// sum along each row and scale rowwise
__kernel void EM_expect_A(__global const float *xi_sum,
                          __global       float *expect_A,
                                   const int N) 
{
        uint gx = get_global_id(0);
        uint gy = get_global_id(1);
        uint lx = get_local_id(0); // col  
        uint ly = get_local_id(1); // row 

        __local float lds[256];

        size_t m =  get_num_groups(0); // number of iterations, equal to the column groups, because A is square 

        int i, col;
        float data;
        size_t offset = gy * N;

        // load 1st time
        data = xi_sum[offset + gx];
        //printf("(%d,%d) \n", gy, gx);

        //#pragma unroll
        for(i = 1 ; i < m ; ++i){
                //col = lx + 16 * i;  
                col = gx + i * TILE;  
                data += xi_sum[offset + col];
        }

        lds[ly*TILE + lx] = data;

        barrier(CLK_LOCAL_MEM_FENCE);

        // sum across rows
        if( gx == 0) // only 16 threads are alive now 
        {
                int start = ly * TILE;
                data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
                        + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
                        + lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
                        + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
                if (data == 0.f) data = 1.f; 
                lds[start] = data;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(i = 0 ; i < m ; ++i){
                col = gx + i * TILE;  
                expect_A[offset + col] = xi_sum[offset + col]/lds[ly * TILE];
        }

}

__kernel void EM_transpose(__global const float *A,
                           __global       float *At,
                                    const int height,
                                    const int width)
{

        __local float lds[272]; // (16 +1) x 16

        // read the matrix tile into shared memory
        uint xIndex = get_group_id(0) * TILE + get_local_id(0);
        uint yIndex = get_group_id(1) * TILE + get_local_id(1);
        uint lidx = get_local_id(0); // col  
        uint lidy = get_local_id(1); // row 

        if((xIndex < width) && (yIndex < height))
        {
                size_t index_in = yIndex * width + xIndex;
                lds[lidy * (TILE + 1) + lidx] = A[index_in];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // write the transposed matrix tile to global memory
        xIndex = get_group_id(1) * TILE + get_local_id(0);
        yIndex = get_group_id(0) * TILE + get_local_id(1);


        if((xIndex < height) && (yIndex < width))
        {
                size_t index_out = yIndex * height + xIndex;
                At[index_out] = lds[lidx * (TILE + 1) + lidy];
        }

}

__kernel void EM_gammastatesum(__global const float *gammaT,
                               __global       float *gamma_state_sum,
                                        const int N,
                                        const int T)
{
        // gammaT :  N x T        
        __local float lds[272]; // 16 x 17 

        uint gx = get_global_id(0);
        uint gy = get_global_id(1);
        uint lx = get_local_id(0); // col  
        uint ly = get_local_id(1); // row 

        size_t m = T / TILE; 

        int i, col;
        float data;
        size_t offset = gy * T;

        // load 1st time
        data = gammaT[offset + gx];

        //#pragma unroll
        for(i = 1 ; i < m ; ++i){
                //col = lx + 16 * i;  
                col = i * TILE + gx;  
                data += gammaT[offset + col];
        }

        lds[ly*(TILE+1) + lx]= data;

        barrier(CLK_LOCAL_MEM_FENCE);

        if( gx == 0) // only 16 threads are alive now 
        {
                int start = ly * (TILE+1);
                data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
                        + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
                        + lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
                        + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
                gamma_state_sum[gy] = data;
        }
}

__kernel void EM_gammaobs(__global const float *observationsT, // D x T
                          __global       float *gamma_obs,
                          __constant     float *bufferT,
                                   const int T)
{
        uint gx = get_global_id(0);// col
        uint gy = get_global_id(1);

        uint id = gy * T + gx;

        // gamma_obs[gy][gx] = observationsT[gy][gx] * bufferT[gx];
        gamma_obs[id] = observationsT[id] * bufferT[gx];

}

__kernel void EM_expectmu(__global const float *gamma_obs, // D x T
                                   const int hs,
                          __global       float *expect_mu, // N x D
                          __constant     float *gamma_state_sumC,
                                   const int T, 
                                   const uint current)
{
        // D x T        
        // row-wise sum 
        __local float lds[272]; // 16 x 16 

        uint gx = get_global_id(0);
        uint gy = get_global_id(1);
        uint lx = get_local_id(0); // col  
        uint ly = get_local_id(1); // row 

        int m = T / TILE;  // devide column T into m TILE-trunks

        int i, col;
        float data;

        uint offset = gy * T;

        // load 1st time
        data = gamma_obs[offset + gx];

        //#pragma unroll
        for(i = 1 ; i < m ; ++i){
                //col = lx + 16 * i;  
                col = i * TILE + gx;  
                data += gamma_obs[offset + col];
        }

        lds[ly*(TILE+1) + lx]= data;

        barrier(CLK_LOCAL_MEM_FENCE);

        if( gx == 0) // only 16 threads are alive now 
        {
                int start = ly * (TILE+1);
                data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
                        + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
                        + lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
                        + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
                expect_mu[current + gy] = data / gamma_state_sumC[hs];
        }

}

__kernel void EM_expectsigma_dev(
                __global const float *gamma_obs,
                __global const float *observations,        
                         const int hs,
                __global       float *expect_sigma_sym,
                __constant     float *gamma_state_sumC,
                __constant     float *expect_mu_state,
                         const int D,
                         const int T)
{
        // C = A x B
        // C , expect_sigma_sym 
        // A , gamma_obs 
        // B , observations

        // (DxT) (TxD) will produce DxD 
        __local float lds_a[72]; // 8 x 9 
        __local float lds_b[72]; // 

        uint lx = get_local_id(0); // col  
        uint ly = get_local_id(1); // row 

        int bx = get_group_id(0);
        int by = get_group_id(1);

        int nx = T / 8;
        int Col =  bx * 8 + lx; // global col index for output
        int Row =  by * 8 + ly; // global row index for output

        float sum = 0.f;        
        int m;

        for ( m = 0; m < nx ; ++m)
        {
                lds_a[ly * 9 + lx] = gamma_obs[Row * T + m * 8 + lx];        
                lds_b[ly * 9 + lx] = observations[(m * 8 + ly) * D + Col];        

                barrier(CLK_LOCAL_MEM_FENCE);

                // matrix mul on LDS
                // a x b
                int kk;
#pragma unroll
                for ( kk = 0; kk < 8; ++kk) 
                {
                        sum += lds_a[ly * 9 + kk] * lds_b[kk * 9 + lx];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

        }

        // sum is the mm result of gamma_obs * obs_t
        // sum * gamma_state_sum(s) - expect_mu(s) * expect_mu(s)'
        expect_sigma_sym[Row * D + Col] = sum / gamma_state_sumC[hs] - expect_mu_state[Row] * expect_mu_state[Col];
}

__kernel void EM_update_expectsigma(
                __global       float *expect_sigma,        
                __global const float *expect_sigma_sym,
                         const int blk_rows,
                         const int width,
                         const uint start)
{
        //__local int2 blkid;
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

        //int2 blkid;

        if(lid_x == 0 && lid_y == 0)
        {
                int i;
                int upper,lower;

                for(i = 2, lower = 0, upper = 1 ; i <= (blk_rows+1) ; i++)
                {
                        if( bn >= lower && bn < upper)
                        {
                                blkid.y = i-2; // rows
                                blkid.x = bn - lower; // cols
                                break;
                        }
                        else
                        {
                                lower = upper;
                                upper = upper + i;
                        }
                }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // find the corresponding global thread index
        size_t gx, gy, gid;
        gx = blkid.x * TILE + lid_x;  // global column index
        gy = blkid.y * TILE + lid_y;
        gid = gy * width + gx;

        size_t gid_sym = gx * width + gy;

        float a = expect_sigma_sym[gid];
        float b = expect_sigma_sym[gid_sym];

        if(gx == gy)
        {
                a = a + 0.01f;
                b = b + 0.01f;
        }

        if( a > b )
        {
                expect_sigma[start + gid] = a;
                expect_sigma[start + gid_sym] = a;
        }
        else
        {
                expect_sigma[start + gid] = b;
                expect_sigma[start + gid_sym] = b;
        }

}
