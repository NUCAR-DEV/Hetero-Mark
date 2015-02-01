// Kernel parameter order : 
//							const parameters, 
//							__global const data, 
//							__global output data

//-----------------------------------------------------------------------------------------------//
// Forward kernels
//-----------------------------------------------------------------------------------------------//
__kernel void FWD_init_alpha(const int    N,
                             __global float *b,
                             __global float *prior,
                             __global float *alpha,
                             __global float *beta)
{
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
	for(i=gid; i<N; i+=gls)
	{
		tidsum += alpha[startpos + i];
	}

	sm[tid] = tidsum;

	barrier(CLK_LOCAL_MEM_FENCE);


	// sum the value from each thread using shared memory
	if(bls >= 512){if(tid < 256) {sm[tid] += sm[tid + 256];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >= 256){if(tid < 128) {sm[tid] += sm[tid + 128];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >= 128){if(tid <  64) {sm[tid] += sm[tid +  64];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  64){if(tid <  32) {sm[tid] += sm[tid +  32];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  32){if(tid <  16) {sm[tid] += sm[tid +  16];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  16){if(tid <   8) {sm[tid] += sm[tid +   8];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   8){if(tid <   4) {sm[tid] += sm[tid +   4];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   4){if(tid <   2) {sm[tid] += sm[tid +   2];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   2){if(tid <   1) {sm[tid] += sm[tid +   1];} barrier(CLK_LOCAL_MEM_FENCE);}


	// element-wise division
	for(i=gid; i<N; i+=gls)
	{
		alpha[startpos + i] /= sm[0];
	}

	if(gid == 0)
	{
		lll[0] += log(sm[0]); 
	}
}

__kernel void TransposeSym (const int N,
                            __local float *sm,
                            __global float *a,
							__global float *aT)
{
	// x : col			y: row
	size_t lx = get_local_id(0);
	size_t ly = get_local_id(1);

	//size_t gx = get_group_id(0) * 16 + lx; 
	//size_t gy = get_group_id(1) * 16 + ly; 
	size_t gx = get_global_id(0); 
	size_t gy = get_global_id(1); 

	//  width, height
	if((gx < N)	&& (gy < N))
	{
		size_t index_in = gy * N + gx;
		sm[ly * 17 + lx] = a[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// transposed block index 
	gx = get_group_id(1) * 16 + lx;
	gy = get_group_id(0) * 16 + ly;

	// height, width
	if((gx < N)	&& (gy < N))
	{
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
								 __global float *alpha)
{
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
	for(i=0; i<iters; ++i)
	{
		int col = i * 16 + lx;
		data += aT[gy * N + col] * constMem[col];	
	}

	sm[ly * 17 + lx] = data;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(gx == 0) // only first column exectues
	{
		int start = ly * 17;	

		data =  sm[start]      + sm[start + 1]  + sm[start + 2]  + sm[start + 3]
			+ sm[start + 4]  + sm[start + 5]  + sm[start + 6]  + sm[start + 7]
			+ sm[start + 8]  + sm[start + 9]  + sm[start + 10] + sm[start + 11]
			+ sm[start + 12] + sm[start + 13] + sm[start + 14] + sm[start + 15];


		alpha[current + gy] = data * b[current + gy];
	}
}


//-----------------------------------------------------------------------------------------------//
// Backward kernels
//-----------------------------------------------------------------------------------------------//
__kernel void BK_BetaB(const int N,
                       const int pos,
                       __global const float *beta,
                       __global const float *b,
                       __global float *betaB)
{
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
							 __global float *beta)
{
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
	for(i=0; i<iters; ++i)
	{
		int col = i * 16 + lx;
		data += a[gy * N + col] * constMem[col];	
	}

	sm[ly * 17 + lx] = data;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(gx == 0) // only first column exectues
	{
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
	for(i=gid; i<N; i+=gls)
	{
		tidsum += beta[current + i];
	}

	sm[tid] = tidsum;

	barrier(CLK_LOCAL_MEM_FENCE);


	// sum the value from each thread using shared memory
	if(bls >= 512){if(tid < 256) {sm[tid] += sm[tid + 256];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >= 256){if(tid < 128) {sm[tid] += sm[tid + 128];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >= 128){if(tid <  64) {sm[tid] += sm[tid +  64];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  64){if(tid <  32) {sm[tid] += sm[tid +  32];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  32){if(tid <  16) {sm[tid] += sm[tid +  16];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=  16){if(tid <   8) {sm[tid] += sm[tid +   8];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   8){if(tid <   4) {sm[tid] += sm[tid +   4];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   4){if(tid <   2) {sm[tid] += sm[tid +   2];} barrier(CLK_LOCAL_MEM_FENCE);}
	if(bls >=   2){if(tid <   1) {sm[tid] += sm[tid +   1];} barrier(CLK_LOCAL_MEM_FENCE);}


	// element-wise division
	for(i=gid; i<N; i+=gls)
	{
		beta[current + i] /= sm[0];
	}
}



/*

// Compute beta * B and alpha * beta
__kernel void EM_betaB_alphabeta(__global const float *beta, 
                                 __global const float *B, 
                                 __global       float *betaB,  
                                 __global const float *alpha,
                                 __global       float *alpha_beta,
                                          const int N,
                                          const int current, 
                                          const int previous)
{
        size_t idx = get_global_id(0);
        if (idx < N) {
                betaB[idx]     = beta[previous + idx] * B[previous + idx];
                alpha_beta[idx] = beta[current + idx] * alpha[current + idx];
        }
}

__kernel void EM_alphabeta_update_gamma(__global const float *alpha_beta, 
                                        __global       float *gamma,
                                        __global const float *ll_d, 
                                                 const int    N, 
                                                 const uint   current)
{
        uint idx = get_global_id(0);
        if (idx < N){
            gamma[current + idx] = alpha_beta[idx] / ll_d[0];
        }
}

// Compute the summation of alpha * beta ( = alpha_beta)
__kernel void EM_sum_alphabeta(__global const float *alpha_beta,
                               __global       float *ll_d,
                                        const int    N,
                               __local        float *sm)
{
    size_t gid = get_global_id(0);  
    size_t lid = get_local_id(0);   
    size_t gls = get_global_size(0);
    size_t bls = get_local_size(0);

    float tidsum = 0.f;
    for(int i=gid; i<N; i+=gls)
    {
        tidsum += alpha_beta[i];    
    }

    sm[gid] = tidsum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // work group reduction
    if(bls >= 512){if(lid < 256) {sm[lid] += sm[lid + 256];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >= 256){if(lid < 128) {sm[lid] += sm[lid + 128];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >= 128){if(lid <  64) {sm[lid] += sm[lid +  64];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >=  64){if(lid <  32) {sm[lid] += sm[lid +  32];} barrier(CLK_LOCAL_MEM_FENCE);}

    // wavefront size for AMD southern islands GPUs is 16
    if(lid < 16)
    {
        if(bls >= 32) {sm[lid] += sm[lid + 16];}    
        if(bls >= 16) {sm[lid] += sm[lid +  8];}    
        if(bls >=  8) {sm[lid] += sm[lid +  4];}    
        if(bls >=  4) {sm[lid] += sm[lid +  2];}    
        if(bls >=  2) {sm[lid] += sm[lid +  1];}    
    }

    if(lid == 0){
        ll_d[0] = sm[0];    
    }
}


__kernel void EM_norm_alphabeta(__global const float *alpha_d,
                                __global const float *beta_d,
                                __global       float *alphabeta_d,
                                __global       float *gamma_d,
                                __local        float *sm,
                                const int current,
                                const int N)
{
    size_t gid = get_global_id(0);  
    size_t lid = get_local_id(0);   
    size_t gls = get_global_size(0);
    size_t bls = get_local_size(0);

    float tidsum = 0.f;
    float tmp;
    for(int i=gid; i<N; i+=gls)
    {
        alphabeta_d[i] = tmp = alpha_d[current+i] * beta_d[current + i];    
        tidsum += tmp; 
    }

    sm[gid] = tidsum;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // work group reduction
    if(bls >= 512){if(lid < 256) {sm[lid] += sm[lid + 256];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >= 256){if(lid < 128) {sm[lid] += sm[lid + 128];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >= 128){if(lid <  64) {sm[lid] += sm[lid +  64];} barrier(CLK_LOCAL_MEM_FENCE);}
    if(bls >=  64){if(lid <  32) {sm[lid] += sm[lid +  32];} barrier(CLK_LOCAL_MEM_FENCE);}

    // wavefront size for AMD southern islands GPUs is 16
    if(lid < 16)
    {
        if(bls >= 32) {sm[lid] += sm[lid + 16];}    
        if(bls >= 16) {sm[lid] += sm[lid +  8];}    
        if(bls >=  8) {sm[lid] += sm[lid +  4];}    
        if(bls >=  4) {sm[lid] += sm[lid +  2];}    
        if(bls >=  2) {sm[lid] += sm[lid +  1];}    
    }

    // sm[0] has the sum
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i=gid; i<N; i+=gls)
    {
        gamma_d[current + i] = alphabeta_d[i] / sm[0];
    }
}



// Compute A. * (alpha * betaB')  
__kernel void EM_A_mul_alphabetaB(__global const float *A, 
                                  __global       float *A_alphabetaB,
                                  __global       float *blk_result,
                                  __constant     float *ConstA,
                                  __constant     float *ConstB, 
                                           const int    N)
{
        size_t lx = get_local_id(0); // col  
        size_t ly = get_local_id(1); // row 

        size_t gx = get_group_id(0) * get_local_size(0) + lx;
        size_t gy = get_group_id(1) * get_local_size(1) + ly;

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
                                       const int   N) 
{
        size_t gx = get_global_id(0);
        size_t gy = get_global_id(1);
        size_t outID = gy * N + gx;
        xi_sum[outID] += A_alphabetaB[outID] / sum;
}



__kernel void EM_expect_A(__global const float *xi_sum_d,
                          __global       float *expt_A_d,
                                   const int N) 
{
        uint gx = get_global_id(0);
        uint lx = get_local_id(0); // col  

        uint gy = get_global_id(1);
        uint ly = get_local_id(1); // row 

        __local float lds[256];

        size_t m =  get_num_groups(0); // number of iterations, equal to the column groups, because A is square 

        int i, col;
        float data;
        size_t offset = gy * N;

        // load 1st time
        data = xi_sum_d[offset + gx];
        //printf("(%d,%d) \n", gy, gx);

        //#pragma unroll
        for(i = 1 ; i < m ; ++i){
                //col = lx + 16 * i;  
                col = gx + i * TILE;  
                data += xi_sum_d[offset + col];
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
                expt_A_d[offset + col] = xi_sum_d[offset + col]/lds[ly * TILE];
        }

}




__kernel void EM_transpose(__global const float *A,
                           __global       float *At,
                                    const int height,
                                    const int width)
{

        __local float lds[272]; // (16 +1) x 16

        // read the matrix tile into shared memory
        size_t  xIndex = get_group_id(0) * TILE + get_local_id(0);
        size_t  yIndex = get_group_id(1) * TILE + get_local_id(1);
        size_t  lidx = get_local_id(0); // col  
        size_t  lidy = get_local_id(1); // row 

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
                          __global       float *expect_mu, // N x D
                          __constant     float *gamma_state_sumC,
                                   const int    hs,
                                   const int    T, 
                                   const       int current)
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
                __global       float *expect_sigma_sym,
                __constant     float *gamma_state_sumC,
                __constant     float *expect_mu_state,
                         const int hs,
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

*/
