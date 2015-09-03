/*!
 * Calculate a FIR filter
 * one work group, and the number of work items is the number of output points
 */

__kernel void FIR(
  __global float * output,
  __global float * coeff,
  __global float * temp_input,
  uint numTap){
#if 1
    uint tid = get_global_id(0);
    uint numData = get_global_size(0);
    uint xid = tid + numTap - 1;

    float sum = 0;
    uint i=0;

    // FIR calculation
    for( i=0; i<numTap; i++ )
    {
        sum = sum + coeff[i] * temp_input[tid + i];
     
    }
    output[tid] = sum;

    // Sync threads
    barrier( CLK_GLOBAL_MEM_FENCE );

    /* fill the history buffer */
    if( tid >= numData - numTap + 1 )
        temp_input[ tid - ( numData - numTap + 1 ) ] = temp_input[xid];

    // Sync threads
    barrier( CLK_GLOBAL_MEM_FENCE );
#endif
}
