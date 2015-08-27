/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research
 * (NUCAR)
 * Group, Northeastern University,
 * http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy
 * of this software and associated documentation files (the
 * "Software"), to deal
 *  with the Software without restriction, including without
 * limitation
 * the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the
 * Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright
 *   notice, this
 *   list of conditions and the following disclaimers. Redistributions
 *   in binary
 *   form must reproduce the above copyright notice, this list of
 *   conditions and
 *   the following disclaimers in the documentation and/or other
 *   materials
 *   provided with the distribution. Neither the names of NUCAR,
 *   Northeastern
 *   University, nor the names of its contributors may be used to
 *   endorse or
 *   promote products derived from this Software without specific
 *   prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *   MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
 *   SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 *   DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *   ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *   OTHER
 *   DEALINGS WITH THE SOFTWARE.
 *
 * Calculating page rank of a webpage i.e. the importance of the webpage
 *
 */

#include "include/pagerank.h"
#include <memory>

PageRank::PageRank() {
  runtime  = clRuntime::getInstance();
  file     = clFile::getInstance();

  platform = runtime->getPlatformID();
  device   = runtime->getDevice();
  context  = runtime->getContext();
  cmdQueue = runtime->getCmdQueue(0);
  workGroupSize = 64;
  maxIter = 10;
}

PageRank::PageRank(std::string fName1, std::string fName2) : PageRank() {
  isVectorGiven = 1;
  fileName1 = fName1;
  fileName2 = fName2;
}

PageRank::PageRank(std::string fName1) : PageRank() {
  isVectorGiven = 0;
  fileName1 = fName1;
}

PageRank::~PageRank() {
  FreeKernel();
  FreeBuffer();
}

void PageRank::InitKernel() {
  // Open kernel file
  file->open("pagerank_Kernels.cl");

  // Create program
  const char *source = file->getSourceChar();
  program = clCreateProgramWithSource(context, 1,
                                      (const char **)&source, NULL, &err);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0x10000,
                          buf,
                          NULL);
    printf("\n%s\n", buf);
    exit(-1);
  }

  // Create program with OpenCL 1.2 support
  err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL1.2", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");
  // Create kernel
  kernel = clCreateKernel(program, "pageRank_kernel", &err);
  checkOpenCLErrors(err, "Failed to create page rank kernel...\n");
}

void PageRank::InitBuffer() {
  InitBufferCpu();
  InitBufferGpu();
}

void PageRank::InitBufferGpu() {
  d_rowOffset = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               sizeof(int)*(nr+1), NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer d_rowOffset...\n");
  d_col = clCreateBuffer(context, CL_MEM_READ_ONLY,
                         sizeof(int)*nnz, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer d_col...\n");
  d_val = clCreateBuffer(context, CL_MEM_READ_ONLY,
                         sizeof(float)*nnz, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer d_val...\n");
  d_vector = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float)*nr, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer d_eigenV...\n");
  d_eigenV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float)*nr, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer d_eigenV...\n");
}

void PageRank::InitBufferCpu() {
  rowOffset = new int[nr + 1];
  col = new int[nnz];
  val = new float[nnz];
  vector = new float[nr];
  eigenV = new float[nr];
}

void PageRank::FreeKernel() {
}

void PageRank::FreeBuffer() {
  csrMatrix.close();
  denseVector.close();
}

void PageRank::FillBuffer() {
  FillBufferCpu();
  FillBufferGpu();
}

void PageRank::FillBufferGpu() {
  err = clEnqueueWriteBuffer(cmdQueue, d_rowOffset, CL_TRUE, 0,
                             sizeof(int)*(nr+1), rowOffset, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to write buffer...\n");
  err = clEnqueueWriteBuffer(cmdQueue, d_col, CL_TRUE, 0,
                             sizeof(int)*nnz, col, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to write buffer...\n");
  err = clEnqueueWriteBuffer(cmdQueue, d_val, CL_TRUE, 0,
                             sizeof(float)*nnz, val, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to write buffer...\n");
  err = clEnqueueWriteBuffer(cmdQueue, d_vector, CL_TRUE, 0,
                             sizeof(float)*nr, vector, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to write buffer...\n");
  // err = clEnqueueWriteBuffer(cmdQueue, d_eigenV, CL_TRUE, 0,
  //                            sizeof(float)*nr, eigenV, NULL, NULL, NULL);
  // checkOpenCLErrors(err, "Failed to write buffer...\n");
}


void PageRank::FillBufferCpu() {
  while (!csrMatrix.eof()) {
    for (int j = 0; j < nr+1; j++)
      csrMatrix >> rowOffset[j];

    for (int j = 0; j < nnz; j++) {
      csrMatrix >> col[j];
    }

    for (int j = 0; j < nnz; j++) {
      csrMatrix >> val[j];
      val[j] = (float)val[j];
    }
  }
  if (isVectorGiven) {
    while (!denseVector.eof()) {
      for (int j = 0; j < nr; j++) {
        denseVector >> vector[j];
        eigenV[j] = 0.0;
      }
    }
  } else {
    for (int j = 0; j < nr; j++) {
      vector[j] = (float)1/(float)nr;
      eigenV[j] = 0.0;
    }
  }
}

void PageRank::ReadCsrMatrix() {
  csrMatrix.open(fileName1);
  if (!csrMatrix.good()) {
    std::cout << "cannot open csr matrix file" << std::endl;
    exit(-1);
  }
  csrMatrix >> nnz >> nr;
}

void PageRank::ReadDenseVector() {
  if (isVectorGiven) {
    denseVector.open(fileName2);
    if (!denseVector.good()) {
      std::cout << "Cannot open dense vector file" << std::endl;
      exit(-1);
    }
  }
}

void PageRank::PrintOutput() {
  std::cout << std::endl << "Eigen Vector: " << std::endl;
  for (int i = 0; i < nr; i++)
    std::cout << eigenV[i] << "\t";
  std::cout << std::endl;
}

void PageRank::Print() {
  std::cout << "nnz: " << nnz << std::endl;
  std::cout << "nr: " << nr << std::endl;
  std::cout << "Row Offset: " << std::endl;
  for (int i = 0; i < nr+1; i++)
    std::cout << rowOffset[i] << "\t";
  std::cout << std::endl << "Columns: " << std::endl;
  for (int i = 0; i < nnz; i++)
    std::cout << col[i] << "\t";
  std::cout << std::endl << "Values: " << std::endl;
  for (int i = 0; i < nnz; i++)
    std::cout << val[i] << "\t";
  std::cout << std::endl << "Vector: " << std::endl;
  for (int i = 0; i < nr; i++)
    std::cout << vector[i] << "\t";
  std::cout << std::endl << "Eigen Vector: " << std::endl;
  for (int i = 0; i < nr; i++)
    std::cout << eigenV[i] << "\t";
  std::cout << std::endl;
}

void PageRank::ExecKernel() {
  global_work_size[0] = nr * workGroupSize;
  local_work_size[0] = workGroupSize;
  int i = 0;
  err = clSetKernelArg(kernel, i++, sizeof(int), &nr);
  err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_rowOffset);
  err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_col);
  err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_val);
  err |= clSetKernelArg(kernel, i++, sizeof(float)*64, NULL);
  checkOpenCLErrors(err, "Failed to setKernelArg...\n");
  for (int j = 0; j < maxIter; j++) {
    if (j % 2 == 0) {
      err |= clSetKernelArg(kernel, i, sizeof(cl_mem), &d_vector);
      err |= clSetKernelArg(kernel, i+1, sizeof(cl_mem), &d_eigenV);
      checkOpenCLErrors(err, "Failed to setKernelArg...\n");
    } else {
      err |= clSetKernelArg(kernel, i, sizeof(cl_mem), &d_eigenV);
      err |= clSetKernelArg(kernel, i+1, sizeof(cl_mem), &d_vector);
      checkOpenCLErrors(err, "Failed to setKernelArg...\n");
    }
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
  }
}

void PageRank::ReadBuffer() {
  if (maxIter % 2 == 1)
    clEnqueueReadBuffer(cmdQueue, d_eigenV, CL_TRUE, 0,
                        sizeof(float)*nr, eigenV, 0, NULL, NULL);
  else
    clEnqueueReadBuffer(cmdQueue, d_vector, CL_TRUE, 0,
                        sizeof(float)*nr, eigenV, 0, NULL, NULL);
}

void PageRank::CpuRun() {
  ReadCsrMatrix();
  ReadDenseVector();
  InitBufferCpu();
  FillBufferCpu();
  for (int i = 0; i < maxIter; i++) {
    PageRankCpu();
    if (i != maxIter - 1) {
      for (int j = 0; j < nr; j++) {
        vector[j] = eigenV[j];
        eigenV[j] = 0.0;
      }
    }
  }
}

float* PageRank::GetEigenV() {
  return eigenV;
}

void PageRank::PageRankCpu() {
  for (int row = 0; row < nr; row++) {
    eigenV[row] = 0;
    float dot = 0;
    int row_start = rowOffset[row];
    int row_end = rowOffset[row+1];

    for (int jj = row_start; jj < row_end; jj++)
      dot += val[jj] * vector[col[jj]];

    eigenV[row] += dot;
  }
}

void PageRank::Run() {
  // Input adjacency matrix from file
  ReadCsrMatrix();
  ReadDenseVector();
  // Initilize the buffer on device
  InitBuffer();
  // Fill in the buffer and transfer them onto device
  FillBuffer();
  // Print();
  // Use a kernel to convert the adajcency matrix to column stocastic matrix

  // The pagerank kernel where SPMV is iteratively called
  InitKernel();
  // Execute the kernel
  ExecKernel();
  // Read the eigen vector back to host memory
  ReadBuffer();
}

void PageRank::Test() {
  ReadCsrMatrix();
  ReadDenseVector();
  InitBufferCpu();
  FillBufferCpu();
  Print();
}
int PageRank::GetLength() {
  return nr;
}
float PageRank::abs(float num) {
  if (num < 0) {
    num = -num;
  }
  return num;
}

int main(int argc, char const *argv[]) {
  uint64_t diff;
  struct timespec start, end;
  if (argc < 2) {
    std::cout << "Usage: pagerank input_matrix [input_vector]" << std::endl;
    exit(-1);
  }
  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  std::unique_ptr<PageRank> pr;
  std::unique_ptr<PageRank> prCpu;
  if (argc == 2) {
    pr.reset(new PageRank(argv[1]));
    prCpu.reset(new PageRank(argv[1]));
    pr->Run();
    prCpu->CpuRun();
  } else if (argc == 3) {
    pr.reset(new PageRank(argv[1], argv[2]));
    prCpu.reset(new PageRank(argv[1], argv[2]));
    pr->Run();
    prCpu->CpuRun();
  }
  float* eigenGpu = pr->GetEigenV();
  float* eigenCpu = prCpu->GetEigenV();
  for (int i = 0; i < pr->GetLength(); i++) {
//  if ( eigenGpu[i] != eigenCpu[i] ) {
    if (pr->abs(eigenGpu[i] - eigenCpu[i]) >= 1e-5) {
      std::cout << "Not Correct!" << std::endl;
      std::cout.precision(20);
      std::cout << pr->abs(eigenGpu[i] - eigenCpu[i]) << std::endl;
      // std::cout << std::abs(1.23f) << std::endl;
      // std::cout << eigenGpu[i] << "\t" << eigenCpu[i] << std::endl;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("Total elapsed time = %llu nanoseconds\n", \
         (long long unsigned int) diff);
  return 0;
}
