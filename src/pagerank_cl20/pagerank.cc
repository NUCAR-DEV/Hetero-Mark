#include "pagerank.h"
#include <memory>

PageRank::PageRank()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0,CL_QUEUE_PROFILING_ENABLE);
	workGroupSize = 64;
	maxIter = 10;
}

PageRank::PageRank(std::string fName1, std::string fName2) : PageRank()
{
	isVectorGiven = 1;
	fileName1 = fName1;
	fileName2 = fName2;
}

PageRank::PageRank(std::string fName1) : PageRank()
{
	isVectorGiven = 0;
	fileName1 = fName1;
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

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
	// Create kernel
	kernel = clCreateKernel(program, "pageRank_kernel", &err);
        checkOpenCLErrors(err, "Failed to create page rank kernel...\n");

}

void PageRank::InitBuffer()
{
	InitBufferCpu();
	InitBufferGpu();
}

void PageRank::InitBufferGpu()
{
}

void PageRank::InitBufferCpu()
{
	rowOffset = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(int)*(nr+1), 0);
	col = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(int)*(nnz), 0);
	val = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(float)*(nnz), 0);
	vector = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(float)*(nr), 0);
	eigenV = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(float)*(nr), 0);
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
	FillBufferCpu();
	FillBufferGpu();
}

void PageRank::FillBufferGpu()
{
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
	if(isVectorGiven) {
		while(!denseVector.eof()) {
			for (int j = 0; j < nr; j++) {
				denseVector >> vector[j];
				eigenV[j] = 0.0;
			}
		}
	}
	else {
		for (int j = 0; j < nr; j++) {
			vector[j] = (float)1/(float)nr;
			eigenV[j] = 0.0;
		}
	}
}
void PageRank::ReadCsrMatrix()
{
	csrMatrix.open(fileName1);
	if(!csrMatrix.good()) {
		std::cout << "cannot open csr matrix file" << std::endl;
		exit(-1);
	}
	csrMatrix >> nnz >> nr; 
}

void PageRank::ReadDenseVector()
{
	if(isVectorGiven) {
		denseVector.open(fileName2);
		if(!denseVector.good()) {
			std::cout << "Cannot open dense vector file" << std::endl;
			exit(-1);
		}
	}
}
void PageRank::PrintOutput()
{
	std::cout << std::endl << "Eigen Vector: " << std::endl;
	for (int i = 0; i < nr; i++)
		std::cout << eigenV[i] << "\t";
	std::cout << std::endl;
}
void PageRank::Print()
{
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

void PageRank::ExecKernel()
{
	global_work_size[0] = nr * workGroupSize;
	local_work_size[0] = workGroupSize;
	int i = 0;
	err = clSetKernelArg(kernel, i++, sizeof(int), &nr);
	err |= clSetKernelArgSVMPointer(kernel, i++, (void*)rowOffset);
	err |= clSetKernelArgSVMPointer(kernel, i++, (void*)col);
	err |= clSetKernelArgSVMPointer(kernel, i++, (void*)val);
	err |= clSetKernelArg(kernel, i++, sizeof(float)*64, NULL);
	checkOpenCLErrors(err, "Failed to setKernelArg...\n");
	for(int j = 0; j < maxIter; j++) {
		if(j % 2 == 0) {
			err |= clSetKernelArgSVMPointer(kernel, i, (void*)vector);
			err |= clSetKernelArgSVMPointer(kernel, i+1, (void*)eigenV);
			checkOpenCLErrors(err, "Failed to setKernelArg...\n");
		} else {
			err |= clSetKernelArgSVMPointer(kernel, i, (void*)eigenV);
			err |= clSetKernelArgSVMPointer(kernel, i+1, (void*)vector);
			checkOpenCLErrors(err, "Failed to setKernelArg...\n");
		}
		err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,	global_work_size, local_work_size, 0, NULL, NULL);
	}
}

void PageRank::ReadBuffer()
{
	if(maxIter % 2 == 0)
		eigenV = vector;
}

void PageRank::CpuRun()
{
	ReadCsrMatrix();
	ReadDenseVector();
	InitBufferCpu();
	FillBufferCpu();
	for(int i = 0; i < maxIter; i++) {
		PageRankCpu();
		if(i != maxIter - 1) {
			for(int j = 0; j < nr; j++) {
				vector[j] = eigenV[j];
				eigenV[j] = 0.0;
			}
		}
	}
}

float* PageRank::GetEigenV()
{
	return eigenV;
}

void PageRank::PageRankCpu()
{
	for(int row = 0; row < nr; row++) {
		eigenV[row] = 0;
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
}
int PageRank::GetLength()
{
	return nr;
}

float PageRank::abs(float num)
{
	if (num < 0) {
		num = -num;
	}
	return num;
}



int main(int argc, char const *argv[])
{
	if (argc < 2) {
		std::cout << "Usage: pagerank input_matrix [input_vector]" << std::endl;
		exit(-1);
	}
	std::unique_ptr<PageRank> pr;
	std::unique_ptr<PageRank> prCpu;
	if (argc == 2) {
		pr.reset(new PageRank(argv[1]));
		prCpu.reset(new PageRank(argv[1]));
		pr->Run();
		prCpu->CpuRun();
	}
	else if (argc == 3) {
		pr.reset(new PageRank(argv[1], argv[2]));
		prCpu.reset(new PageRank(argv[1], argv[2]));
		pr->Run();
		prCpu->CpuRun();
	}
	float* eigenGpu = pr->GetEigenV();
	float* eigenCpu = prCpu->GetEigenV();
	for(int i = 0; i < pr->GetLength(); i++) {
//		if( eigenGpu[i] != eigenCpu[i] ) {
		if( pr->abs(eigenGpu[i] - eigenCpu[i]) >= 1e-5 ) {
			std::cout << "Not Correct!" << std::endl;
			std::cout.precision(20);
			std::cout << pr->abs(eigenGpu[i] - eigenCpu[i]) << std::endl;
			//std::cout << std::abs(1.23f) << std::endl;
			//std::cout << eigenGpu[i] << "\t" << eigenCpu[i] << std::endl;
		}
	}
	return 0;
}
