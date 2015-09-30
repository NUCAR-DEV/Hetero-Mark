# Hetero-Mark

A Benchmark Suite for Heterogeneous System Computation

## Description

The program takes plaintext as input and encrypts it using a given
encryption key. Our implementation uses a key size of 256 bits. The
AES algorithm is comprised of many rounds, that ultimately turn
plaintext into cipher-text. Each round has multiple processing steps
that include AddRoundKey, SubBytes, ShiftRows and MixColumns. Key bits
 must be expanded using a precise key expansion schedule.

## Usage

====== Hetero-Mark AES Benchmarks (OpenCL 1.2) ======
This benchmarks runs the AES Algorithm.

Help[bool]: -h --help (default = false)
  Dump help information

InputFile[string]: -i --input (default = in.txt)
  The input file to be encrypted

Keyfile[string]: -k --keyfile (default = key.txt)
  The file containing the key in hex format

Mode[string]: -m --mode (default = h)
  Mode of input parsing, 'h' interpretes the data in hex format, 'a' interpretes the data as ASCII

OutFile[string]: -o --output (default = out.bin)
  The file where the encrypted output should be written