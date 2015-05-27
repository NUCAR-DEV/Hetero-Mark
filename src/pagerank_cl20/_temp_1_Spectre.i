# 1 "/tmp/OCL7569T5.cl"
# 1 "/tmp/OCL7569T5.cl" 1
# 1 "<built-in>" 1
# 1 "<built-in>" 3
#define __clang__ 1
#define __ENDIAN_LITTLE__ 1
#define __SPIR64 1
#define __SPIR64__ 1
#define __STDC__ 1
#define __STDC_HOSTED__ 1
#define __STDC_VERSION__ 199901L
#define __OPENCL_C_VERSION__ 200
#define __OPENCL_VERSION__ 200

# 1 "<command line>" 1
#define CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE 120795904
#define FP_FAST_FMA 1
#define cl_khr_fp64 1
#define cl_amd_fp64 1
#define cl_khr_global_int32_base_atomics 1
#define cl_khr_global_int32_extended_atomics 1
#define cl_khr_local_int32_base_atomics 1
#define cl_khr_local_int32_extended_atomics 1
#define cl_khr_int64_base_atomics 1
#define cl_khr_int64_extended_atomics 1
#define cl_khr_3d_image_writes 1
#define cl_khr_byte_addressable_store 1
#define cl_khr_gl_sharing 1
#define cl_ext_atomic_counters_32 1
#define cl_amd_device_attribute_query 1
#define cl_amd_vec3 1
#define cl_amd_printf 1
#define cl_amd_media_ops 1
#define cl_amd_media_ops2 1
#define cl_amd_popcnt 1
#define cl_khr_image2d_from_buffer 1
#define cl_khr_spir 1
#define cl_khr_subgroups 1
#define cl_khr_gl_event 1
#define cl_khr_depth_images 1
#define __IMAGE_SUPPORT__ 1
#define __AMD__ 1
# 1 "<built-in>" 2
# 1 "/tmp/OCL7569T5.cl" 2






__kernel void pageRank_kernel(int num_rows,
              __global int* rowOffset,
              __global int* col,
              __global float* val,
              __local float *vals,
              __global float* x,
              __global float* y)
{
  int thread_id = get_global_id(0);
  int local_id = get_local_id(0);
  int warp_id = thread_id / 64;
  int lane = thread_id & (64 - 1);
  int row = warp_id;

  if (row < num_rows) {
    y[row] = 0.0;
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
