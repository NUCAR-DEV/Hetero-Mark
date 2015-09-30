# Hetero-Mark

A Benchmark Suite for Heterogeneous System Computation

## Description

An IIR filter requires less processing
power than an FIR filter for
the same design requirements. The implementation decomposes
IIR into multiple parallel second-order IIR filters to achieve better
performance.

## Usage

====== Hetero-Mark IIR Benchmarks (OpenCL 1.2) ======
This benchmarks runs the parallel IIR for multi-channel case.

Help[bool]: -h --help (default = false)
  Dump help information

Length[int]: -l --length (default = 256)
  Length of input