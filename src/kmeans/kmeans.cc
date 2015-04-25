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

void KMEANS::Read()
{
	
	
	
}




int main( int argc, char** argv) 
{

	std::unique_ptr<KMEANS> kmeans(new KMEANS);

	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", 
		BLOCK_SIZE, BLOCK_SIZE2);

	//setup(argc, argv);
	//shutdown();

	return 0;
}
