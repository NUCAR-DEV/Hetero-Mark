#pragma OPENCL EXTENSION cl_amd_printf : enable 

#define ROWS 256  // num of parallel subfilters

__kernel void IIR(const int len, 
                     const float c, 
                     __constant float *nsec, 
                     __constant float *dsec, 
                     __local float *sm,
                     __constant float *x, 
                     __global float *y) 
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int bls = get_local_size(0);

	float2 u;
	u.x =0.f;
	u.y =0.f;

	float new_u;

	int i, j;
	for(i=0; i<len; i+=ROWS)
	{
		// load one block of data at a time
		sm[lid]	= x[i + lid];

		barrier(CLK_LOCAL_MEM_FENCE);	

		// go through each input in local memory
		for(j=0; j<ROWS; j++)
		{
			new_u = sm[j] - dot(dsec[lid], u); 	
			// shuffle one to the right in u
			u.y = u.x;
			u.x = new_u;

			float suby = dot(nsec[lid], u);
			float blk_sum = work_group_reduce_add(suby);

			// output	
			if(lid == 0)
			{
				y[get_group_id(0) * len + i + j] = blk_sum + c * sm[j];
			}
		} // end of block input
	} // end of input
}
