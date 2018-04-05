# Hetero-Mark

A Benchmark Suite for collaborative CPU-GPU computing.

## Prerequisite

### OpenCL Environment

* [OpenCL](http://support.amd.com/en-us/kb-articles/Pages/AMD-Radeon-GPU-PRO-Linux-Beta-Driver%E2%80%93Release-Notes.aspx) - The OpenCL driver

We use to use the FGLRX driver on Ubuntu 14.04, which supports OpenCL 2.0.
Since AMD stopped the support for the FGLRX driver on ubuntu 16.04, we
switched to the AMDGPU-pro driver. Currently, OpenCL 2.0 benchmarks are
not supported on this platform.

### HSA Environment

* [ROCm](https://github.com/RadeonOpenCompute/ROCm) - Radeon Open Compute

Although ROCm 1.6 platform does not fully support OpenCL 2.0, all the
features we have been using in the benchmark suite is supported by ROCm
1.6.

### OpenCV Library

The Background Extraction benchmark will use OpenCV for video decoding and
encoding. The benchmark suite will detect if your system has OpenCV
installed or not. If OpenCV libraries are not found, CMAKE will skip
compiling the BE benchmarks.

We use the following command to install OpenCV libraries.

```bash
sudo apt install libopencv-dev
```

## Applications

Hetero-Mark is designed to model the workloads that is similar to real
world applications, where the major part of the application is written in
general purpose programming languages, while only a small, performance
critical portion is written using GPU-accelerated libraries. So for each
benchmark, we provide a base class that provides platform independent
functionalities, such as input data loading, result verification. For each
GPU programming method (such as CUDA, HC, HIP), we extend the base class
with a sub-class and impletment the "Run" method.

Since the base classes are platform independent, we use plain pointers for
input and output data. Each benchmark will have to read from plain
pointers and finally write the result into other plain pointers. We
believe this behavior is closer to real world senario, since most
programmers do not carry a platform specific memory management system to
the whole application and usually will only use GPU program as a library.
This also suggests that the benchmarking time considers the data copy time
between the CPU and the GPU memory.

All the benchmarks has a verification process where the GPU result is
compared with the CPU result. Although we report the execution time of the
verification process, the time is not meant to compare the CPU performance
to GPU performance. The verification process can be very useful if the
benchmark runs in simulators or if the validity of the platform is under
evaluation.

* Advanced Encryption Standard (AES) - The program takes plaintext as
input and encrypts it using a given encryption key. Our implementation
uses a key size of 256 bits. The AES algorithm is comprised of many
rounds, that ultimately turnplaintext into cipher-text. Each round has
multiple processing stepsthat include AddRoundKey, SubBytes, ShiftRows
and MixColumns. Key bits must be expanded using a precise key expansion

* Background Extraction (BE) - An useful algorithm in video and image
processing, background extraction algorithms usually create a background
model based on static components of the frame. Our implementation uses
a Running Gaussian Average, and takes an input video file and extracts
the background of that video.

* Black Scholes (BS) - The Black–Scholes or Black–Scholes–Merton model is
a mathematical model of a financial market containing derivative
investment instruments. From the model, one can deduce the Black–Scholes
formula, which gives a theoretical estimate of the price of
European-style options.

* Binary Search Tree Insertion (BSTI) - Binary Search Tree is a useful
data structure for its balanced insertion and in-order accessing
performance, but rearranging an array into a binary search tree is
usually time consuming. The GPU can help with inserting the nodes of
a binary search tree in parallel, using one thread to insert one node.
However, as the output is a irregular tree structure, we need to let the
CPU and the GPU collaborate under the Co-Contributing pattern.

* Color Histogramming (CH) - Color Histogramming is a popular method in
image processing to divide the color space into groups, and counts the
number of pixels in a picture that fall into each group. The
implementation of Color Histogramming is divided into two phases. In the
first phase the GPU kernel scans the whole image and each GPU thread
covers a small portion of the image. Each thread stores the histogram
information of the pixels it has scanned in a region of the private
memory that is dedicated to that thread. In the second phase, each GPU
thread takes the histogram produced in the first phase and adds it to an
output histogram using atomic operations.

* Force Directed Edge Bundling (FDEB) - Force Directed Edge Bundling is
a graph-based data visualization algorithm that helps readers identify
patterns in a complex graph. The algorithm models a spring between each
pair of the edges and calculates the forces applied to points on each
edge. Then each point moves a certain distance towards the direction of
the combined force. 

* Evolutionary Programming (EP) - Evolutionary Programming solves
optimization problems using an approach that mimics the natural
selection process. In our benchmark implementation, we use Evolutionary
Programming to solve a non-convex optimization problem.

* Finite Impulse Response (FIR) - FIR filter produces an impulse response
of finite duration. The impulse response is the response to any finite
length input. The FIR filtering program is designed to have the host
send array data to the FIR kernel on the OpenCL device. Then the FIR
filter is calculated on the device, and the result is transferred back
to the host.

* Gene Alignment (GA) - Gene Alignment algorithms are used to answer
questions about specific gene sequences (e.g., “CATGCATG”) that occur in
the human gene sequence. Our implementation uses a modified version of
the Basic Local Alignment Search Tool (BLAST).

* K-Nearest Neighbors (KNN) - Given a large number of labeled training
samples in a multi-dimensional feature space, the K- Nearest Neighbors
(KNN) algorithm takes a query point and searches for the K training
samples that are close to that point. Using a majority vote approach,
the KNN algorithm can categorize the query point with the label that
appears the most number of times in the selected K training samples. 

* KMeans (KM) - k-means clustering is a method of vector quantization,
originally from signal processing, that is popular for cluster analysis
in data mining. k-means clustering aims to partition n observations into
k clusters in which each observation belongs to the cluster with the
nearest mean, serving as a prototype of the cluster. In this
implementation, we have varied the number of objects of 34 features and
put them into 5 clusters. The input file contains features and
attributes.

* Page Rank (PR) - PageRank is an algorithm used by Google Search to rank
websites in their search engine results. It is a link analysis algorithm
and it assigns a numerical weighting to each element of a hyperlinked
set of documents, such as the World Wide Web, with the purpose of
"measuring" its relative importance within the set. So the computations
are representatives of graph based applications.

## Compiling the code

### OpenCL

Use the following commands to compile the OpenCL benchmarks.

```bash
mkdir build
cd build
cmake -DCOMPILE_OPENCL12 ../
make
```

If OpenCL is properly configured in your system, the command above will
use the system default compiler to compile OpenCL benchmarks.

### HCC Compilation

Use the following commands to compile the HCC benchmarks

```bash
mkdir build
cd build
CXX=hcc cmake ../
make
```

This command will also use the HCC compiler to compile the OpenCL
benchmarks.

### CUDA Compilation

Use the following commands to compile CUDA benchmarks. Make sure your
system has NVCC compiler installed.

```bash
mkdir build
cd build
cmake -DCOMPILE_CUDA=On ../
make
```

### HIP Compilation

Use the following commands to compile HIP benchmarks.

```bash
mkdir build
cd build
cmake -DCOMPILE_HIP=On ../
make
```

HIP works for both CUDA platform and the ROCm platform.

## Run the code

The executables are in the build folder under
`Hetero-Mark/build/src/<application name>/<environment>` if you follow the
default compile guide, where `<application name>` is the name of the
application, such as, fir, be, bs etc and replace `<environment>` for
cl12, cl20, cuda or hc.

The executables support the following arguments:

* `-t` is for timing information
* `-v` is for cpu verification 
* `-q` is for suppressing the output

All benchmark executables has a `-h` option.
The help documentation of each benchmark explains how to use the benchmark
and what parameter is needed.

## Input data

### Download standard input data

```bash
You can download the standard data from the following url
https://drive.google.com/file/d/1IItjFFUIfANgrUUI7jebNS9rfSEe32lZ/view?usp=sharing".

### Generate your own input data

* To generate custom data in `data` folder

    * AES - Generates the input file and keys for AES. For keys, only 16-byte is
      allowed.

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

## Known Issue

* The HIP version of Background Extraction (BE) benchmark cannot compile
on CUDA platform currently. This is due to that fact that the HIP cmake
configuration uses NVCC as program linker which NVCC cannot handle the
linking to OpenCV properly. We have been actively resolving the problem
with ROCm developers. See
[#120](https://github.com/ROCm-Developer-Tools/HIP/issues/120). 

## Development guide

Please raise issues on the Github page if you have any questions or
problems using the benchmark suite.

We accept pull requests on github if you want to contribute to the
benchmark suite. If you have any question or problem with HeteroMark,
please file an issue in our github repo.

Hetero-mark follows [google c++ coding
style](https://google.github.io/styleguide/cppguide.html) in header files
and source files. We also use `make check` to lint the source code using
the `clang-format` tool and `cpplint` tool. 

## Citation

* Yifan Sun, Saoni Mukherjee, Trinayan Baruah, Shi Dong, Julian Gutierrez,
Prannoy Mohan and David Kaeli, "[Evaluating Performance Tradeoffs on the
Radeon Open Compute Platform]()" 2018 IEEE International Symposium on
Performance Analysis of Systems and Software (ISPASS), Belfast, 2018. 

* Yifan Sun, Xiang Gong, Amir Kavyan Ziabari, Leiming Yu, Xiangyu Li,
Saoni Mukherjee, Carter McCardwell, Alejandro Villegas, David Kaeli,
"[Hetero-mark, a benchmark suite for CPU-GPU collaborative
computing](https://www.researchgate.net/profile/Yifan_Sun12/publication/309206460_Hetero-Mark_A_Benchmark_Suite_for_CPU-GPU_Collaborative_Computing/links/5805891408aef179365e7183.pdf)"
2016 IEEE International Symposium on Workload Characterization (IISWC),
Providence, RI, USA, 2016, pp. 1-10.

* Saoni Mukherjee, Yifan Sun, Paul Blinzer, Amir Kavyan Ziabari and David
Kaeli, "[A comprehensive performance analysis of HSA and OpenCL
2.0](https://www.researchgate.net/profile/David_Kaeli2/publication/303772633_A_comprehensive_performance_analysis_of_HSA_and_OpenCL_20/links/575464ab08ae02ac128112bb.pdf)"
2016 IEEE International Symposium on Performance Analysis of Systems and
Software (ISPASS), Uppsala, 2016, pp. 183-193.

* Saoni Mukherjee, Xiang Gong, Leiming Yu, Carter McCardwell, Yash
Ukidave, Tuan Dao, Fanny Nina Paravecino, and David Kaeli. [Exploring the
Features of OpenCL
2.0](http://www1.coe.neu.edu/~saoni/files/Mukherjee_IWOCL_2015.pdf) The
International Workshop on OpenCL (IWOCL), 2015.
