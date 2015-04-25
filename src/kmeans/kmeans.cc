#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <cassert>

#include <clUtil.h>

using namespace std;

#include "kmeans.h"

KMEANS::KMEANS()
{	
}

KMEANS::~KMEANS()
{	
	CleanUpKernels();
	CleanUpBuffers();
}

void KMEANS::CleanUpKernels()
{
	
}

void KMEANS::CleanUpBuffers()
{
	
}

void KMEANS::Usage(char *argv0)
{
	const char *help =
		"\nUsage: %s [switches] -i filename\n\n"
		"    -i filename      :file containing data to be clustered\n"		
		"    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
		"    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
		"    -t threshold     :threshold value                       [default=0.001]\n"
		"    -l nloops        :iteration for each number of clusters [default=1]\n"
		"    -b               :input file is in binary format\n"
		"    -r               :calculate RMSE                        [default=off]\n"
		"    -o               :output cluster center coordinates     [default=off]\n";
	fprintf(stderr, help, argv0);
	exit(-1);
}

void KMEANS::Read(int argc, char **argv)
{
	// ------------------------- command line options -----------------------//
	int     opt;                                                                
	extern char   *optarg;                                                      
	isBinaryFile = 0;                                                       
	threshold = 0.001;          // default value
	max_nclusters=5;            // default value
	min_nclusters=5;            // default value
	isRMSE = 0;                                                             
	isOutput = 0;                                            
	nloops = 1;                 // default value

	char    line[1024];                                                         
	ssize_t ret; // add return value for read

	float  *buf;                                                                
	npoints = 0;                                                            
	nfeatures = 0;                                                          

	best_nclusters = 0;                                                     

	int i, j;

	// obtain command line arguments and change appropriate options
	while ( (opt=getopt(argc,argv,"i:t:m:n:l:bro"))!= EOF) {                    
		switch (opt) {                                                          
			case 'i': filename=optarg;                                          
					  break;                                                      
			case 'b': isBinaryFile = 1;                                         
					  break;                                                      
			case 't': threshold=atof(optarg);                                   
					  break;                                                      
			case 'm': max_nclusters = atoi(optarg);                             
					  break;                                                      
			case 'n': min_nclusters = atoi(optarg);                             
					  break;                                                      
			case 'r': isRMSE = 1;                                               
					  break;                                                      
			case 'o': isOutput = 1;                                             
					  break;                                                      
			case 'l': nloops = atoi(optarg);                                    
					  break;                                                      
			case '?': Usage(argv[0]);                                           
					  break;                                                      
			default: Usage(argv[0]);                                            
					 break;                                                      
		}                                                                       
	}                                                                       

	if (filename == 0) Usage(argv[0]);  
	
	// ============== I/O begin ==============//

	//io_timing = omp_get_wtime();
	if (isBinaryFile) 
	{	//Binary file input
		int infile;
		if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
			fprintf(stderr, "Error: no such file (%s)\n", filename);
			exit(1);
		}

		ret = read(infile, &npoints, sizeof(int));

		if (ret == -1) {
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);
		}

		ret = read(infile, &nfeatures, sizeof(int));        
		if (ret == -1) {
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);
		}

		// allocate space for features[][] and read attributes of all objects
		// defined in header file
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		feature    = (float**)malloc(npoints*          sizeof(float*));
		feature[0] = (float*) malloc(npoints*nfeatures*sizeof(float));

		// fixme: svm buffer
		for (i=1; i<npoints; i++)
			feature[i] = feature[i-1] + nfeatures;

		ret = read(infile, buf, npoints*nfeatures*sizeof(float));

		if (ret == -1) {
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);
		}

		close(infile);
	}
	else 
	{
		FILE *infile;
		if ((infile = fopen(filename, "r")) == NULL) {
			fprintf(stderr, "Error: no such file (%s)\n", filename);
			exit(1);
		}		

		while (fgets(line, 1024, infile) != NULL) {
			if (strtok(line, " \t\n") != 0)
				npoints++;			
		}


		rewind(infile);

		while (fgets(line, 1024, infile) != NULL) {
			if (strtok(line, " \t\n") != 0) {
				// ignore the id (first attribute): nfeatures = 1;
				while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
				break;
			}
		}        

		// allocate space for features[] and read attributes of all objects
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		feature    = (float**)malloc(npoints*          sizeof(float*));
		feature[0] = (float*) malloc(npoints*nfeatures*sizeof(float));

		// fixme : svm buffer
		for (i=1; i<npoints; i++)
			feature[i] = feature[i-1] + nfeatures;

		rewind(infile);

		i = 0;

		while (fgets(line, 1024, infile) != NULL) {

			if (strtok(line, " \t\n") == NULL) continue;            

			for (j=0; j<nfeatures; j++) {
				buf[i] = atof(strtok(NULL, " ,\t\n"));             
				i++;
			}            
		}

		fclose(infile);
	}

	//io_timing = omp_get_wtime() - io_timing;

	printf("\nI/O completed\n");
	printf("\nNumber of objects: %d\n", npoints);
	printf("Number of features: %d\n", nfeatures);
	
	// error check for clusters
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", 
				min_nclusters, npoints);
		exit(0);
	}

	// now features holds 2-dimensional array of features //
	memcpy(feature[0], buf, npoints*nfeatures*sizeof(float));
	free(buf);
}

void KMEANS::CL_initialize()
{

	runtime    = clRuntime::getInstance();
	// OpenCL objects get from clRuntime class
	platform   = runtime->getPlatformID();
	context    = runtime->getContext();
	device     = runtime->getDevice();
	cmd_queue  = runtime->getCmdQueue(0);

}

void KMEANS::CL_build_program()
{
	cl_int err;
	// Helper to read kernel file
	file = clFile::getInstance();
	file->open("kmeans.cl");

	const char *source = file->getSourceChar();
	prog = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
	checkOpenCLErrors(err, "Failed to create Program with source...\n");


	// Create program with OpenCL 2.0 support
	err = clBuildProgram(prog, 0, NULL, "-I ./ -cl-std=CL2.0", NULL, NULL);
	checkOpenCLErrors(err, "Failed to build program...\n");

}

void KMEANS::CL_create_kernels()
{
	cl_int err;
	// Create kernels
	kernel_s = clCreateKernel(prog, "kmeans_kernel_c", &err);
	checkOpenCLErrors(err, "Failed to create kmeans_kernel_c");

	kernel2 = clCreateKernel(prog, "kmeans_swap", &err);
	checkOpenCLErrors(err, "Failed to create kernel kmeans_swap");
}

void KMEANS::CL_create_buffers(int nclusters)
{
	cl_int err;

	// Create buffers
	d_feature = clCreateBuffer(context, 
	                           CL_MEM_READ_WRITE, 
							   npoints * nfeatures * sizeof(float), 
							   NULL, 
							   &err);
	checkOpenCLErrors(err, "clCreateBuffer d_feature failed");

	d_feature_swap = clCreateBuffer(context, 
	                                CL_MEM_READ_WRITE, 
									npoints * nfeatures * sizeof(float), 
									NULL, 
									&err);
	checkOpenCLErrors(err, "clCreateBuffer d_feature_swap failed");


	d_membership = clCreateBuffer(context, 
	                              CL_MEM_READ_WRITE, 
								  npoints * sizeof(int), 
								  NULL, 
								  &err);
	checkOpenCLErrors(err, "clCreateBuffer d_membership failed");


	d_cluster = clCreateBuffer(context, 
	                           CL_MEM_READ_WRITE, 
							   nclusters * nfeatures  * sizeof(float), 
							   NULL, 
							   &err);
	checkOpenCLErrors(err, "clCreateBuffer d_cluster failed");
}


void KMEANS::Swap_features()
{
	cl_int err;

	// fixme
	err = clEnqueueWriteBuffer(cmd_queue, 
			d_feature, 
			1, 
			0, 
			npoints * nfeatures * sizeof(float), 
			feature[0], 
			0, 0, 0);
	checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer d_feature");

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &npoints);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &nfeatures);	

	size_t global_work     = (size_t) npoints;
	size_t local_work_size = BLOCK_SIZE;

	if(global_work % local_work_size != 0)
		global_work = (global_work / local_work_size + 1) * local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, 
	                             kernel2, 
								 1, 
								 NULL, 
								 &global_work, 
								 &local_work_size, 
								 0, 0, 0);
	checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel()");
}
	
	
	
	
	
	
void KMEANS::Clustering()
{
	cluster_centres = NULL;
	index =0;			    // number of iteration to reach the best RMSE

	// fixme
	membership = (int*) malloc(npoints * sizeof(int));



	CL_initialize();
	CL_build_program();
	CL_create_kernels();


	min_rmse_ref = FLT_MAX;		

	int	nclusters;			    // number of clusters
	// sweep k from min to max_nclusters to find the best number of clusters
	for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
	{
		// cannot have more clusters than points
		if (nclusters > npoints) 
			break;	

		// allocate device memory, invert data array
		CL_create_buffers(nclusters);

		Swap_features();

/*
		// iterate nloops times for each number of clusters //
		for(i = 0; i < nloops; i++)
		{
			// initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) //
			tmp_cluster_centres = kmeans_clustering(features,
					nfeatures,
					npoints,
					nclusters,
					threshold,
					membership);

			if (*cluster_centres) {
				free((*cluster_centres)[0]);
				free(*cluster_centres);
			}

			*cluster_centres = tmp_cluster_centres;

			// find the number of clusters with the best RMSE //
			if(isRMSE)
			{
				rmse = rms_err(features,
						nfeatures,
						npoints,
						tmp_cluster_centres,
						nclusters);

				if(rmse < min_rmse_ref){
					min_rmse_ref = rmse;		//update reference min RMSE
					*min_rmse = min_rmse_ref;	//update return min RMSE
					*best_nclusters = nclusters;	//update optimum number of clusters
					index = i;			//update number of iteration to reach best RMSE
				}
			}			
		}
*/

		//deallocateMemory(); // free device memory
	}

	// free(membership);

}

void KMEANS::Run(int argc, char **argv)                                         
{                                                                               
	// ----------------- Read input file and allocate features --------------//
	Read(argc, argv);                                                         

	// ----------------- Clustering -------------------------- --------------//
	//cluster_timing = omp_get_wtime();		// Total clustering time
	Clustering();
	//cluster_timing = omp_get_wtime() - cluster_timing;	
		                                                                                
}    



int main( int argc, char** argv) 
{

	std::unique_ptr<KMEANS> kmeans(new KMEANS);

	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", 
		BLOCK_SIZE, BLOCK_SIZE2);

	kmeans->Run(argc, argv);
	//setup(argc, argv);
	//shutdown();

	return 0;
}
