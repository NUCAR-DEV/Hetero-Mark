// IIR KERNEL
#define ROWS 256  // num of parallel subfilters

//__kernel void __shfl_xor ()

__kernel void IIR (__global float *x, __global int len, __global float c, __global float *y, 
					__global float2 *NSEC, __global float2 *DSEC)
{
	extern __global float sm[]; // originally shared
	float *sp = &sm[ROWS]; // ROWS = 256

	int tid = get_global_id(0); // CUDA: threadIdx OpenCL: get_global_id(0)

	// & 0x20
	int lane_id = tid % 32; // warp size 32 for +3.5 device
	int warp_id = tid / 32;

	int ii, jj, kk;

	float2 u = (float2)(0.0f); // CUDA specific
	float unew;
	float y0;

	// block size : ROWS
	// each thread fetch input x to shared memory
	for(ii=0; ii<len; ii+=ROWS)
	{
		sm[tid] = x[tid + ii];	

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// go through each x in shared memory 
		for(jj=0; jj<ROWS; jj++)	
		{
			unew = sm[jj] - dot(u, DSEC[tid]);				
			u = (float2)(unew, u.x); // make_float2 CUDA specific
			y0 = dot(u, NSEC[tid]);
		
			// sum v across current block
			#pragma unroll
			for(kk=1; kk<32; kk<<=1)  
			{
				y0 += __shfl_xor(y0, kk, 32); 
			}

			if(lane_id == 0)
			{
				sp[warp_id] = y0;
			}

			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			if(blockSize == 256 && warp_id == 0)
			{
				if(lane_id < 8)
				{
					float warp_sum = sp[lane_id];

					warp_sum += __shfl_xor(warp_sum, 1, 32);  // ? 32
					warp_sum += __shfl_xor(warp_sum, 2, 32);  // ? 32
					warp_sum += __shfl_xor(warp_sum, 4, 32);  // ? 32

					if(lane_id == 0){
						// channel starting postion: blockId.x * len
						uint gid = mul24(get_group_id(0) , len) + ii + jj;
						y[gid] = warp_sum + sm[jj] * c;	 
					}
				}
			}
		}
	}
}

