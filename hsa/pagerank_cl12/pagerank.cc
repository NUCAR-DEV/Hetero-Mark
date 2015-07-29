#include "pagerank.h"
#include <memory>

PageRank::PageRank()
{
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
    	FreeBuffer();
}

void PageRank::InitBuffer()
{
	rowOffset= new int[nr + 1]; 
	col = new int[nnz];
	val = new float[nnz];
	vector = new float[nr];
	eigenV = new float[nr];
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
			val[j] = (float)val[j];
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

	SNK_INIT_LPARM(lparm, 0);
	lparm->ldims[0] = local_work_size[0];
	lparm->gdims[0] = nr * workGroupSize;

	for (int j = 0; j < maxIter; j++) {
		if (j % 2 == 0) {
			pageRank_kernel(nr, rowOffset, col, val, 
					sizeof(float)*64, vector, eigenV,
					lparm);
		} else {
			pageRank_kernel(nr, rowOffset, col, val, 
					sizeof(float)*64, eigenV, vector,
					lparm);
		}
	}
}

void PageRank::CpuRun()
{
	ReadCsrMatrix();
	ReadDenseVector();
	InitBuffer();
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

	//Execute the kernel
	ExecKernel();
}

void PageRank::Test()
{
	ReadCsrMatrix();
	ReadDenseVector();
	InitBuffer();
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
	clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("Total elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	return 0;
}
