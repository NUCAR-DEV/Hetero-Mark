#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define Nb 4
#define Nr 14
#define Nk 8

void SubBytes(uchar* input, uchar* s) {
  input[0] = s[input[0]];
  input[1] = s[input[1]];
  input[2] = s[input[2]];
  input[3] = s[input[3]];
  input[4] = s[input[4]];
  input[5] = s[input[5]];
  input[6] = s[input[6]];
  input[7] = s[input[7]];
  input[8] = s[input[8]];
  input[9] = s[input[9]];
  input[10] = s[input[10]];
  input[11] = s[input[11]];
  input[12] = s[input[12]];
  input[13] = s[input[13]];
  input[14] = s[input[14]];
  input[15] = s[input[15]];
}

void MixColumns(uchar* arr) {
  uchar a[16] = {0};
  uchar b[16] = {0};
  uchar h[16] = {0};
  int i = 0;

  for (uchar c = 0; c < 16; c++) {
    a[c] = arr[c];
    h[c] = a[c] & 0x80;
    b[c] = h[c] == 0x80 ? (a[c] << 1) ^ 0x1b : a[c] << 1;
  }

  arr[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
  arr[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
  arr[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
  arr[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];

  arr[4] = b[4] ^ a[7] ^ a[6] ^ b[5] ^ a[5];
  arr[5] = b[5] ^ a[4] ^ a[7] ^ b[6] ^ a[6];
  arr[6] = b[6] ^ a[5] ^ a[4] ^ b[7] ^ a[7];
  arr[7] = b[7] ^ a[6] ^ a[5] ^ b[4] ^ a[4];

  arr[8] = b[8] ^ a[11] ^ a[10] ^ b[11] ^ a[9];
  arr[9] = b[9] ^ a[8] ^ a[11] ^ b[10] ^ a[10];
  arr[10] = b[10] ^ a[9] ^ a[8] ^ b[9] ^ a[11];
  arr[11] = b[11] ^ a[10] ^ a[9] ^ b[8] ^ a[8];

  arr[12] = b[12] ^ a[15] ^ a[14] ^ b[13] ^ a[13];
  arr[13] = b[13] ^ a[12] ^ a[15] ^ b[14] ^ a[14];
  arr[14] = b[14] ^ a[13] ^ a[12] ^ b[15] ^ a[15];
  arr[15] = b[15] ^ a[14] ^ a[13] ^ b[12] ^ a[12];

}

void ShiftRows(uchar* input) {
  uchar state[16];
  state[0] = input[0];
  state[1] = input[5];
  state[2] = input[10];
  state[3] = input[15];
  state[4] = input[4];
  state[5] = input[9];
  state[6] = input[14];
  state[7] = input[3];
  state[8] = input[8];
  state[9] = input[13];
  state[10] = input[2];
  state[11] = input[7];
  state[12] = input[12];
  state[13] = input[1];
  state[14] = input[6];
  state[15] = input[11];

  input[0] = state[0];
  input[1] = state[1];
  input[2] = state[2];
  input[3] = state[3];
  input[4] = state[4];
  input[5] = state[5];
  input[6] = state[6];
  input[7] = state[7];
  input[8] = state[8];
  input[9] = state[9];
  input[10] = state[10];
  input[11] = state[11];
  input[12] = state[12];
  input[13] = state[13];
  input[14] = state[14];
  input[15] = state[15];
}

void AddRoundKey(uchar* state, __global uchar* expanded_key, int offset) {
  uchar *bytes = expanded_key + offset;

  // The reversed bytes order per 4-bytes matches the endianess of the host.
  state[0] ^= bytes[3];
  state[1] ^= bytes[2];
  state[2] ^= bytes[1];
  state[3] ^= bytes[0];
  state[4] ^= bytes[7];
  state[5] ^= bytes[6];
  state[6] ^= bytes[5];
  state[7] ^= bytes[4];
  state[8] ^= bytes[11];
  state[9] ^= bytes[10];
  state[10] ^= bytes[9];
  state[11] ^= bytes[8];
  state[12] ^= bytes[15];
  state[13] ^= bytes[14];
  state[14] ^= bytes[13];
  state[15] ^= bytes[12];
}

__kernel void Encrypt(__global uchar* input, 
                      __global uchar* expanded_key,
                      __global uchar* s) {
  uchar state[16];

  int tid = get_global_id(0);

  for (int i = 0; i < 16; i++) {
    state[i] = input[tid * 16 + i];
  }

  AddRoundKey(state, expanded_key, 0);
  SubBytes(state, s);
  ShiftRows(state);
  MixColumns(state);

  /*
  for (int i = 1; i < 14; i++) {
    SubBytes(state, s);
    ShiftRows(state);
    MixColumns(state);
    AddRoundKey(state, expanded_key, i * 16);
  }

  SubBytes(state, s);
  ShiftRows(state);
  AddRoundKey(state, expanded_key, 14 * 16);
  */


  for (int i = 0; i < 16; i++) {
    input[tid * 16 + i] = state[i];
  }
}
