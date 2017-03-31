# Hetero-Mark
A Benchmark Suite for Heterogeneous System Computation

## Prerequisite

### OpenCL Environment
* [OpenCL 2.0](http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx) - The OpenCL 2.0 driver

### HSA Environment
* [ROCm](https://github.com/RadeonOpenCompute/ROCm) - Radeon Open Compute
* [CLOC & SNACK](https://github.com/HSAFoundation/CLOC) - OpenCL to HSAIL compiler and snack runtime.
    Make sure that snack.sh is available in system path

## Applications
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

* Black Scholes - The Black–Scholes or Black–Scholes–Merton model is a mathematical
model of a financial market containing derivative investment instruments. From the model,
one can deduce the Black–Scholes formula, which gives a theoretical estimate of
the price of European-style options.

## Compiling the code

Use the following commands to compile the benchmarks

```bash
    mkdir build
    cd build
    cmake ../
    make
```
The previous commands will automitically detect your system environment and compile either 
OpenCL benchmarks or the HSA benchmarks. To compile the HC++ benchmarks, you need to have 
hcc available in your system and replace the `cmake ../` line with 
```bash
    CXX=hcc cmake -DCOMPILE_HCC=On ../
```

## Run the code
The executables are in the build folder if you follow the default compile guide.

All benchmark executables has a `-h` option.
The help documentation of each benchmark explains how to use the benchmark and what parameter is needed.


## Input data

### Download standard input data

```bash
./download_data.sh
```

After executing the bash script, you will have a `data` folder in the project
root directory.

### Generate your own input data
* To generate custom data in `data` folder

  * AES - Generates the input file and keys for AES. For keys, only 16-byte is
    allowed. Usage:
    ``` bash
    ./datagen <num_bytes> > file.data
    ```

  * Gene-alignment - Generates the input file for Gene Alignment. The target
    sequence length should be much shorter than the query sequence length.
    ``` bash
    python data_gen.py <target_sequence_len> <query_sequence_len>
    ```

  * KMeans - It generates the input file for KMeans. Usage:

    ``` bash
    g++ datagen.cpp -o datagen
    ./datagen <numObjects> [ <numFeatures> ] [-f]
    ```

  * PageRank - It generates the input matrix for PageRank. Usage:

    ``` bash
    python datagen.py
    ```

## Development guide

We accept pull requests on github if you want to contribute to the benchmark suite.
If you have any question or problem with HeteroMark, please file an issue in our github repo.

Hetero-mark follows [google c++ coding style](https://google.github.io/styleguide/cppguide.html) in header files and source files.
