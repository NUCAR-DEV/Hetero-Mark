__kernel void accessOnce(
  unsigned long length, 
  __global uint *in, 
  __global uint *out) {
  for (unsigned long i = 0; i < length; i++) {
    out[0] = in[i];
  }
}

__kernel void accessTwice(
  unsigned long length, 
  __global uint *in, 
  __global uint *out) {
  for (unsigned long i = 0; i < length; i++) {
    out[0] = in[i];
    out[0] = in[i];
  }
}
