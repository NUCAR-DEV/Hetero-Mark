# Hetero-Mark
A Benchmark Suite for Heterogeneous System Computation

# Requirements
* [g++](https://gcc.gnu.org/onlinedocs/gcc-3.3.6/gcc/G_002b_002b-and-GCC.html) - The GNU C++ compiler
* [OpenCL 2.0](http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx) - The OpenCL 2.0 driver
* [HSA Driver](https://github.com/HSAFoundation/HSA-Drivers-Linux-AMD) - HSA driver
* [CLOC & SNACK](https://github.com/HSAFoundation/CLOC) - OpenCL to HSAIL compiler and snack runtime.
    Make sure that snack.sh is available in system path

####Applications
The suite is in development. All outputs show the time it takes to run
the application without overheads such as data transfer time etc.

* Advanced Encryption Standard (AES) - The program takes plaintext as input and encrypts it using a given
encryption key. Our implementation uses a key size of 256 bits. The
AES algorithm is comprised of many rounds, that ultimately turn
plaintext into cipher-text. Each round has multiple processing steps
that include AddRoundKey, SubBytes, ShiftRows and MixColumns. Key bits
 must be expanded using a precise key expansion schedule.

* Finite Impulse Response (FIR) - FIR filter produces an impulse response of finite duration. The impulse
 response is the response to any finite length input. The FIR filtering
 program is designed to have the host send array data to the FIR kernel
 on the OpenCL device. Then the FIR filter is calculated on the device,
 and the result is transferred back to the host.

* Hidden Markov Model (HMM) - HMM is a static Markov model that can generate probabilistic meaning
 without knowing the hidden states. The implementation
targets isolated word recognition. In order to achieve the
best performance on the GPU device, we express the data-level
and thread-level parallelism in the HMM algorithm.

* Infinite Impulse Response (IIR) - An IIR filter requires less processing
power than an FIR filter for
the same design requirements. The implementation decomposes
IIR into multiple parallel second-order IIR filters to achieve better
performance.

* KMeans - k-means clustering is a method of vector quantization, originally from
 signal processing, that is popular for cluster analysis in data mining.
 k-means clustering aims to partition n observations into k clusters in
 which each observation belongs to the cluster with the nearest mean,
 serving as a prototype of the cluster. In this implementation, we have
 varied the number of objects of 34 features and put them into 5 clusters.
 The input file contains features and attributes.

* Page Rank - PageRank is an algorithm used by Google Search to rank websites in their
 search engine results. It is a link analysis algorithm and it assigns a
 numerical weighting to each element of a hyperlinked set of documents,
 such as the World Wide Web, with the purpose of "measuring" its relative
 importance within the set. So the computations are representatives of graph
 based applications.

* Shallow Water - Shallow water is a physics simulation engine that depicts complex
 behavior of fluids, wave modeling for interactive systems. It predicts
 matters of practical interest, e.g. internal tides in strait of Gibraltar.

####Compiling the code

Use the following commands to compile the benchmarks

`mkdir build`

`cd build`

`cmake ../`

`make`

### Input data

#### Download standard input data
Standard input is provided for data dependent benchmark such as K-means. 
Cloning Hetero-Mark repository will not download the standard input data. 
Download the standard input data with the following commands:

`git lfs fetch`

You may need to install the Git extension for versioning large files. 
Instructions can be found [here](https://git-lfs.github.com/)

#### Generate your own input data
* To generate custom data in `data` folder
  * AES - It generates custom size plain text file. Usage: `./<exec> <file size>` 
  * KMeans - It generates the input file for KMeans. Usage:

    `g++ KMeans_datagen.cpp -o KMeans_datagen`

    `./KMeans_gen_dataset.sh`

  * PageRank - It generates the input matrix for PageRank. Usage: `python PageRank_generateCsrMatrix.py`

####Note
If error happens during compilation, but you want to compile other benchmarks, use `make -i` to ignore errors. 

####Development guide

Hetero-mark follows [google c++ coding style](https://google.github.io/styleguide/cppguide.html) in header files and source files. 
