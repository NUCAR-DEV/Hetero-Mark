#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <cassert>
#include <clUtil.h>

#include "kmeans.h"
using namespace std;

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
	int		opt;
	extern char   *optarg;
	char   *filename = 0;
	float  *buf;
	char	line[1024];
	int	isBinaryFile = 0;

	float	threshold = 0.001;		/* default value */
	int	max_nclusters=5;		/* default value */
	int	min_nclusters=5;		/* default value */
	int	best_nclusters = 0;
	int	nfeatures = 0;
	int	npoints = 0;
	float	len;

	float **features;
	float **cluster_centres=NULL;
	int	i, j, index;
	int	nloops = 1;				/* default value */

	int	isRMSE = 0;		
	float	rmse;

	int	isOutput = 0;
	//float	cluster_timing, io_timing;
}


void KMEANS::Run(int argc, char **argv)
{
	//Read(argc, argv);	
	
}




int main( int argc, char** argv) 
{

	std::unique_ptr<KMEANS> kmeans(new KMEANS);

	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", 
		BLOCK_SIZE, BLOCK_SIZE2);

	// kmeans->Run(argc, argv);

	//setup(argc, argv);
	//shutdown();

	return 0;
}
