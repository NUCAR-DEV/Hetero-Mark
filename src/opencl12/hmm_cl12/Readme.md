## HMM

HMM is a static Markov model that can generate probabilistic meaning
 without knowing the hidden states. The implementation
targets isolated word recognition. In order to achieve the
best performance on the GPU device, we express the data-level
and thread-level parallelism in the HMM algorithm.

## Usage

====== Hetero-Mark HMM Benchmarks (OpenCL 1.2) ======
This benchmark runs the Hidden Markov Model.

Help[bool]: -h --help (default = false)
  Dump help information

HiddenStates[int]: -s --states (default = 16)
  Number of hidden states
