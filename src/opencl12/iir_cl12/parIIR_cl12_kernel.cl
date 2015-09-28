#pragma OPENCL EXTENSION cl_amd_printf : enable 

#define ROWS 256  // num of parallel subfilters

__kernel void ParIIR(const int len, 
    const float c, 
    __constant float2 *nsec, 
    __constant float2 *dsec, 
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
  for(i=0; i<len; i+=256)
  {
    // load one block of data at a time
    sm[lid]	= x[i + lid];

    barrier(CLK_LOCAL_MEM_FENCE);	

    // go through each input in local memory
    for(j=0; j<256; j++)
    {
      new_u = sm[j] - dot(dsec[lid], u); 	
      // shuffle one to the right in u
      u.y = u.x;
      u.x = new_u;

      sm[256 + lid] = dot(nsec[lid], u);

      barrier(CLK_LOCAL_MEM_FENCE);	

      int pos = 256 + lid;
      // sum the value from each thread using shared memory
      if(bls >= 256) {
        if(lid < 128) { sm[pos] += sm[pos + 128];}
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >= 128) {
        if(lid <  64) {sm[pos] += sm[pos +  64];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=  64) {
        if(lid <  32) {sm[pos] += sm[pos +  32];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=  32) {
        if(lid <  16) {sm[pos] += sm[pos +  16];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=  16) {
        if(lid <  8) {sm[pos] += sm[pos +   8];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=   8) {
        if(lid <   4) {sm[pos] += sm[pos +   4];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=   4) {
        if(lid <   2) {sm[pos] += sm[pos +   2];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      if(bls >=   2) {
        if(lid <   1) {sm[pos] += sm[pos +   1];} 
        barrier(CLK_LOCAL_MEM_FENCE);}

      // output	
      if(lid == 0)
      {
        y[get_group_id(0) * len + i + j] = sm[256] + c * sm[j];
      }

      barrier(CLK_LOCAL_MEM_FENCE);	
    }
  } // end of input
}
