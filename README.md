# Hetero-Mark

A Benchmark Suite for Heterogeneous System Computation

## Applications in the suite

The suite is in development. All outputs show the time it takes to run
the application without the data transfer time etc.

### Advanced Encryption Standard (AES)

The program takes plaintext as input and encrypts it using a given
encryption key. Our implementation uses a key size of 256 bits.
The AES algorithm is comprised of
many rounds, that ultimately turn
plaintext into cipher-text. Each round has multiple processing
steps that include AddRoundKey, SubBytes, ShiftRows and
MixColumns. Key bits must be expanded using a precise key
expansion schedule.

#### Usage

    <exec> <mode> <plain text file> <keyfile> <encrypted text file>
    mode is either h or a

### Finite Impulse Response (FIR)

FIR filter produces an impulse
response of finite duration [6]. The impulse response is the
response to any finite length input. The FIR filtering program
is designed to have the host send array data to the FIR kernel
on the OpenCL device. Then the FIR filter is calculated on
the device, and the result is transferred back to the host.

#### Usage

    ./<exec> <numBlocks> <numData>

### Hidden Markov Model

HMM is a static Markov model
that can generate probabilistic meaning without knowing the
hidden states. The implementation
targets isolated word recognition. In order to achieve the
best performance on the GPU device, we express the data-level
and thread-level parallelism in the HMM algorithm.

#### Usage
     ./<exec> <number of hidden states>

### Infinite Impulse Response (IIR)

An IIR filter requires less processing
power than an FIR filter for
the same design requirements. The implementation decomposes
IIR into multiple parallel second-order IIR filters to achieve better
performance.

#### Usage

     ./<exec> <length of the filter>

### KMeans

k-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. In this implementation, we have varied the number of objects of 34 features and put them into 5 clusters. The input file contains features and attributes.

#### Usage
     ./<exec> [switches] -i <input file>

	-i filename      :file containing data to be clustered
	-m max_nclusters :maximum no. of clusters allowed   [default=5]
	-n min_nclusters :minimum no. of clusters allowed   [default=5]
	-t threshold     :threshold value                   [default=0.001]
	-l nloops        :iteration for each no. of cluster [default=1]
	-b               :input file is in binary format
	-r               :calculate RMSE                    [default=off]
	-o               :output cluster center coordinates [default=off]

### Page Rank

PageRank is an algorithm used by Google Search to rank websites in their search engine results. It is a link analysis algorithm and it assigns a numerical weighting to each element of a hyperlinked set of documents, such as the World Wide Web, with the purpose of "measuring" its relative importance within the set. So the computations are representatives of graph based applications.

#### Usage
     ./<exec> <input matrix>

### Shallow Water

Shallow water is a physics simulation engine that depicts complex behavior of fluids, wave modeling for interactive systems. It predicts matters of practical interest, e.g. internal tides in strait of Gibraltar.

#### Usage
     ./<exec> <dimension in X axis> <dimension in Y axis>


## Compilation

    To compile without debug
        cmake .

    To compile with debug
        cmake -DCMAKE_BUILD_TYPE=Debug .


## Execution

    To run each application, the executables are in
        <app folder>/bin/x86_64/Release/

## Development guide

    The skeleton code for new benchmark is available
    in src/template directory.

    Make sure to add new benchmark dir to CmakeList.txt
    file in src/, otherwise new benchmark won't be
    compiled with others.

## Credits

Northeastern University Computer Architecture Research Group,
Northeastern University
http://www.ece.neu.edu/groups/nucar/




