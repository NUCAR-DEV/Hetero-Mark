## FIR

FIR filter produces an impulse response of finite duration. The impulse
 response is the response to any finite length input. The FIR filtering
 program is designed to have the host send array data to the FIR kernel
 on the OpenCL device. Then the FIR filter is calculated on the device,
 and the result is transferred back to the host.

## Usage

====== Hetero-Mark FIR Benchmarks (OpenCL 1.2) ======
This benchmarks runs the FIR-Filter Algorithm.

Help[bool]: -h --help (default = false)
  Dump help information

NumBlocks[int]: -b --blocks (default = 100)
  Number of test blocks

NumData[int]: -d --data (default = 1000)
  Number of data samples
