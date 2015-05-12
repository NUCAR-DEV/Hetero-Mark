//indices is col
//ptr is rowOffset
//data is the vals
//vals are temporary
//x is the input vector

__kernel void spmv_kernel(int num_rows,
              __global int* rowOffset,
              __global int* col,
              __global float* val,
              __global float* x,
              __global float* y,
              __local float *vals)
{
  int thread_id = get_global_id(0);
  int local_id = get_local_id(0);
  int warp_id = thread_id / 64;
  int lane = thread_id & (64 - 1);
  int row = warp_id;

  if (row < num_rows) {
    int row_A_start = rowOffset[row];
    int row_A_end = rowOffset[row + 1];

    vals[local_id] = 0;
    for(int jj = row_A_start + lane; jj < row_A_end; jj += 64)
      vals[local_id] += val[jj] * x[col[jj]];

    if(lane < 32) vals[local_id] += vals[local_id + 32];
    if(lane < 16) vals[local_id] += vals[local_id + 16];
    if(lane < 8) vals[local_id] += vals[local_id + 8];
    if(lane < 4) vals[local_id] += vals[local_id + 4];
    if(lane < 2) vals[local_id] += vals[local_id + 2];
    if(lane < 1) vals[local_id] += vals[local_id + 1];
    if(lane == 0)
      y[row] += vals[local_id];
  }
}
