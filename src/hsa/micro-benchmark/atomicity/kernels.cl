__kernel void atomicityTest(
  unsigned long length, 
  __global int *memory) {
  for (unsigned long i = 0; i < length; i++) {
    memory[i] = i;
  }
}

