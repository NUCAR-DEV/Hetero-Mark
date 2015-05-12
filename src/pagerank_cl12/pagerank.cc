#include "pagerank.h"
#include <memory>

PageRank::PageRank()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);
	workGroupSize = 64;
}

PageRank::PageRank(string fileName1, string fileName2)
{
	PageRank();
	this->fileName1 = fileName1;
	this->fileName2 = fileName2;
}

PageRank::~PageRank()
{
	FreeKernel();
    	FreeBuffer();
}

void PageRank::InitKernel()
{
        // Open kernel file
        file->open("pagerank_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        if (err != CL_SUCCESS)
        {
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

void PageRank::InitBuffer()
{
	rowOffset= new int[nr + 1]; 
	col = new int[nnz];
	val = new float[nnz];
	eigenV = new float[nr];
	vector = new float[nr];
	d_rowOffset = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*(nr+1), NULL, &err);
        checkOpenCLErrors(err, "Failed to create buffer d_rowOffset...\n");
	d_col = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*nnz, NULL, &err);
        checkOpenCLErrors(err, "Failed to create buffer d_col...\n");
	d_val = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nnz, NULL, &err);
        checkOpenCLErrors(err, "Failed to create buffer d_val...\n");
	d_eigenV = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nnz, NULL, &err);
        checkOpenCLErrors(err, "Failed to create buffer d_eigenV...\n");
}

void PageRank::InitBufferCpu()
{
	rowOffset= new int[nr + 1]; 
	col = new int[nnz];
	val = new float[nnz];
	eigenV = new float[nr];
	vector = new float[nr];
}

void PageRank::FreeKernel()
{
        
}

void PageRank::FreeBuffer()
{
	csrMatrix.close();
	denseVector.close();
}

void PageRank::FillBuffer()
{
	while(!csrMatrix.eof()) {
		for (int j = 0; j < nr+1; j++)
			csrMatrix >> rowOffset[j];

		for (int j = 0; j < nnz; j++) {
			csrMatrix >> col[j];
		}   

		for (int j = 0; j < nnz; j++) {
			csrMatrix >> val[j];
		}   
	}
	while(!denseVector.eof()) {
		for (int j = 0; j < nr; j++)
			denseVector >> vector[j];
	}
	memset(eigenV, 0.0, sizeof(eigenV));

	err = clEnqueueWriteBuffer(cmdQueue, d_rowOffset, CL_TRUE, 0, sizeof(int)*(nr+1), rowOffset, NULL, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmdQueue, d_col, CL_TRUE, 0, sizeof(int)*nnz, col, NULL, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmdQueue, d_val, CL_TRUE, 0, sizeof(float)*nnz, val, NULL, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmdQueue, d_vector, CL_TRUE, 0, sizeof(float)*nr, vector, NULL, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmdQueue, d_eigenV, CL_TRUE, 0, sizeof(float)*nr, eigenV, NULL, NULL, NULL);
	checkOpenCLErrors(err, "Failed to write buffer...\n");
}
	

void PageRank::FillBufferCpu()
{
	while(!csrMatrix.eof()) {
		for (int j = 0; j < nr+1; j++)
			csrMatrix >> rowOffset[j];

		for (int j = 0; j < nnz; j++) {
			csrMatrix >> col[j];
		}   

		for (int j = 0; j < nnz; j++) {
			csrMatrix >> val[j];
		}   
	}
	while(!denseVector.eof()) {
		for (int j = 0; j < nr; j++)
			denseVector >> vector[j];
	}
	memset(eigenV, 0.0, sizeof(eigenV));
}
void PageRank::ReadCsrMatrix()
{
	csrMatrix.open(fileName1);
	if(!csrMatrix.good()) {
		cout << "cannot open csr matrix file" << endl;
		exit(-1);
	}
	csrMatrix >> nnz >> nr; 
}

void PageRank::ReadDenseVector()
{
	denseVector.open(fileName2);
	if(!denseVector.good()) {
		cout << "Cannot open dense vector file" << endl;
		exit(-1);
	}
}
void PageRank::Print()
{
	cout << "nnz: " << nnz << endl;
	cout << "nr: " << nr << endl;
	cout << "Row Offset: " << endl;
	for (int i = 0; i < nr+1; i++)
		cout << rowOffset[i] << "\t";
	cout << endl << "Columns: " << endl;
	for (int i = 0; i < nnz; i++)
		cout << col[i] << "\t";
	cout << endl << "Values: " << endl;
	for (int i = 0; i < nnz; i++)
		cout << val[i] << "\t";
	cout << endl << "Vector: " << endl;
	for (int i = 0; i < nr; i++)
		cout << vector[i] << "\t";
	cout << endl << "Eigen Vector: " << endl;
	for (int i = 0; i < nr; i++)
		cout << eigenV[i] << "\t";
	cout << endl;
}	

void PageRank::ExecKernel()
{
	int i = 0;
	err = clSetKernelArg(kernel, i++, sizeof(int), &nr);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_rowOffset);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_col);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_val);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_vector);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &d_eigenV);
	err |= clSetKernelArg(kernel, i++, sizeof(float)*64, NULL);
	checkOpenCLErrors(err, "Failed to setKernelArg...\n");
	global_work_size[0] = nr * workGroupSize;
	local_work_size[0] = workGroupSize;
	err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,	global_work_size, local_work_size, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to execute NDRange kernel\n");
}

void PageRank::ReadBuffer()
{
	clEnqueueReadBuffer(cmdQueue, d_eigenV, CL_TRUE, 0, sizeof(float)*nr, eigenV, NULL, NULL, NULL);
}

void PageRank::CpuRun()
{
	ReadCsrMatrix();
	InitBufferCpu();
	FillBufferCpu();
	PageRankCpu();
}

float* PageRank::GetEigenV()
{
	return eigenV;
}

void PageRank::PageRankCpu()
{
	for(int row = 0; row < nr; row++) {
		float dot = 0;
		int row_start = rowOffset[row];
		int row_end = rowOffset[row+1];
		
		for(int jj = row_start; jj < row_end; jj++)
			dot += val[jj] * vector[col[jj]];

		eigenV[row] += dot;
	}
}
	
	
void PageRank::Run()
{
	//Input adjacency matrix from file
	ReadCsrMatrix();	
	ReadDenseVector();
	//Initilize the buffer on device
	InitBuffer();
	//Fill in the buffer and transfer them onto device
	FillBuffer();
	//Print();
	//Use a kernel to convert the adajcency matrix to column stocastic matrix

	//The pagerank kernel where SPMV is iteratively called
	InitKernel();
	//Execute the kernel
	ExecKernel();
	//Read the eigen vector back to host memory
	ReadBuffer();
}

void PageRank::Test()
{
	ReadCsrMatrix();
	ReadDenseVector();
	InitBufferCpu();
	FillBufferCpu();
	Print();
	PageRankCpu();
}



int main(int argc, char const *argv[])
{
	if (argc < 3) {
		std::cout << "Usage: pagerank input_matrix input_vector" << endl;
		exit(-1);
	}
	std::unique_ptr<PageRank> pr(new PageRank(argv[1], argv[2]));
	pr->Test();
	//pr->Run();
	return 0;
}
