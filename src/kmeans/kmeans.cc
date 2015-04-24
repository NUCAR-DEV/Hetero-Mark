#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <cassert>

#include <clUtil.h>

using namespace std;

#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
	double gettime() {
		struct timeval t;
		gettimeofday(&t,NULL);
		return t.tv_sec+t.tv_usec*1e-6;
	}
#endif


#ifdef NV 
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
     #define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
     #define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE2 RD_WG_SIZE
#else
     #define BLOCK_SIZE2 256
#endif


extern "C"
{
	#include "kmeans.h"
}















int main( int argc, char** argv) 
{
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", 
		BLOCK_SIZE, BLOCK_SIZE2);

	setup(argc, argv);
	shutdown();
}

int setup(int argc, char **argv) {
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
	
	ssize_t ret; /*add return value for read*/

	/* obtain command line arguments and change appropriate options */
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
			case '?': usage(argv[0]);
					break;
			default: usage(argv[0]);
					break;
		}
    	}

	if (filename == 0) usage(argv[0]);

	//-----------------------------------------------------------------------//
	// OpenCL initialization                                                    
	int use_gpu = 1;                                                            
	if(initialize(use_gpu)) return -1;       

	bool svmCoarseGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_COARSE);
	bool svmFineGrainAvail = clRuntime::getInstance()->isSVMavail(SVM_FINE);

	// Need at least coarse grain
	if (!svmCoarseGrainAvail)
	{
		printf("SVM coarse grain support unavailable\n");
		exit(-1);
	}

	// device memory:	float *feature
	//                  float *clusters
	//                  int   *membership
	//                  float *feature_swap
	//-----------------------------------------------------------------------//

		
	/* ============== I/O begin ==============*/
	/* get nfeatures and npoints */
	//io_timing = omp_get_wtime();

	if (isBinaryFile) {		//Binary file input
		int infile;
		if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
			fprintf(stderr, "Error: no such file (%s)\n", filename);
			exit(1);
		}
		ret = read(infile, &npoints,   sizeof(int));
		if (ret == -1)
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);

		ret = read(infile, &nfeatures, sizeof(int));        
		if (ret == -1)
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);

		/* allocate space for features[][] and read attributes of all objects */
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		features    = (float**)malloc(npoints*          sizeof(float*));
		features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
		for (i=1; i<npoints; i++)
			features[i] = features[i-1] + nfeatures;

		ret = read(infile, buf, npoints*nfeatures*sizeof(float));
		if (ret == -1)
			fprintf(stderr, "Error: failed to read, info: %s.%d\n", 
					__FILE__, __LINE__);

		close(infile);
	}
	else {
		FILE *infile;
		if ((infile = fopen(filename, "r")) == NULL) {
			fprintf(stderr, "Error: no such file (%s)\n", filename);
			exit(1);
		}		
		while (fgets(line, 1024, infile) != NULL)
			if (strtok(line, " \t\n") != 0)
				npoints++;			
		rewind(infile);
		while (fgets(line, 1024, infile) != NULL) {
			if (strtok(line, " \t\n") != 0) {
				/* ignore the id (first attribute): nfeatures = 1; */
				while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
				break;
			}
		}        

		/* allocate space for features[] and read attributes of all objects */
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		features    = (float**)malloc(npoints*          sizeof(float*));
		features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
		for (i=1; i<npoints; i++)
			features[i] = features[i-1] + nfeatures;
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
	/* ============== I/O end ==============*/

	// error check for clusters
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", 
			min_nclusters, npoints);
		exit(0);
	}


	/* seed for future random number generator */	
	srand(7);  

	/* now features holds 2-dimensional array of features */
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float));
	free(buf);

	/* ======================= core of the clustering ===================*/

	//cluster_timing = omp_get_wtime();		/* Total clustering time */
	cluster_centres = NULL;
	index = cluster(npoints,	/* number of data points */
			nfeatures,	/* number of features for each point */
			features,	/* array: [npoints][nfeatures] */
			min_nclusters,	/* range of min to max number of clusters */
			max_nclusters,
			threshold,	/* loop termination factor */
			&best_nclusters,/* return: number between min and max */
			&cluster_centres,/* return: [best_nclusters][nfeatures] */  
			&rmse,		/* Root Mean Squared Error */
			isRMSE,		/* calculate RMSE */
			nloops);	/* number of iteration for each number of clusters */		

	//cluster_timing = omp_get_wtime() - cluster_timing;


	/* =============== Command Line Output =============== */

	/* cluster center coordinates
	   :displayed only for when k=1*/
	if((min_nclusters == max_nclusters) && (isOutput == 1)) {
		printf("\n================= Centroid Coordinates =================\n");
		for(i = 0; i < max_nclusters; i++){
			printf("%d:", i);
			for(j = 0; j < nfeatures; j++){
				printf(" %.2f", cluster_centres[i][j]);
			}
			printf("\n\n");
		}
	}
	
	len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

	printf("Number of Iteration: %d\n", nloops);
	//printf("Time for I/O: %.5fsec\n", io_timing);
	//printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);
	
	if(min_nclusters != max_nclusters){
		if(nloops != 1){	
			//range of k, multiple iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
		else{
			//range of k, single iteration
			//printf("Average Clustering Time: %fsec\n",
			//		cluster_timing / len);
			printf("Best number of clusters is %d\n", best_nclusters);				
		}
	}
	else{
		if(nloops != 1){
			// single k, multiple iteration
			//printf("Average Clustering Time: %.5fsec\n",
			//		cluster_timing / nloops);
			if(isRMSE) {// if calculated RMSE
			printf("Number of trials to approach the best RMSE of %.3f is %d\n", 
				rmse, index + 1);
			}
		}
		else{	
			// single k, single iteration				
			if(isRMSE){
				// if calculated RMSE
				printf("Root Mean Squared Error: %.3f\n", rmse);
			}
		}
	}
	

	/* free up memory */
	free(features[0]);
	free(features);    

	return(0);
}

void usage(char *argv0) {
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

int initialize(int use_gpu)
{
	// Init OCL context
	runtime    = clRuntime::getInstance();
	// OpenCL objects get from clRuntime class
	platform   = runtime->getPlatformID();
	context    = runtime->getContext();
	device     = runtime->getDevice();
	cmd_queue  = runtime->getCmdQueue(0);
	return 0;
}

int cluster(int      npoints,			/* number of data points */
            int      nfeatures,			/* number of attributes for each point */
            float  **features,			/* array: [npoints][nfeatures] */                  
            int      min_nclusters,		/* range of min to max number of clusters */
			int	     max_nclusters,
            float    threshold,			/* loop terminating factor */
            int     *best_nclusters,		/* out: number between min and max with lowest RMSE */
            float ***cluster_centres,		/* out: [best_nclusters][nfeatures] */
	        float    *min_rmse,			/* out: minimum RMSE */
	        int	     isRMSE,			/* calculate RMSE */
	        int	     nloops			/* number of iteration for each number of clusters */ 
			)
{    
	int	nclusters;			/* number of clusters k */	
	int	index =0;			/* number of iteration to reach the best RMSE */
	int	rmse;				/* RMSE for each clustering */
	int    *membership;			/* which cluster a data point belongs to */
	float **tmp_cluster_centres;		/* hold coordinates of cluster centers */
	int	i;

	// here svm
	/* allocate memory for membership */
	membership = (int*) malloc(npoints * sizeof(int));

	/* sweep k from min to max_nclusters to find the best number of clusters */
	for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
	{
		if (nclusters > npoints) break;	/* cannot have more clusters than points */

		/* allocate device memory, invert data array (@ kmeans_cuda.cu) */
		allocate(npoints, nfeatures, nclusters, features);


		/* iterate nloops times for each number of clusters */
		for(i = 0; i < nloops; i++)
		{
			/* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
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

			/* find the number of clusters with the best RMSE */
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

		deallocateMemory(); /* free device memory (@ kmeans_cuda.cu) */
	}

	free(membership);

	return index;
}


int allocate(int n_points, int n_features, int n_clusters, float **feature)
{
	cl_int err;

	file->open("kmeans.cl");

	// Create program
	const char *source = file->getSourceChar();

	cl_program prog = clCreateProgramWithSource(context, 1, 
			(const char **)&source, NULL, &err);
	checkOpenCLErrors(err, "Failed to create Program with source...\n");

	// Create program with OpenCL 2.0 support
	err = clBuildProgram(prog, 0, NULL, "-I ./ -cl-std=CL2.0", NULL, NULL);
	checkOpenCLErrors(err, "Failed to build program...\n");

	// Create kernels
	kernel_s = clCreateKernel(prog, "kmeans_kernel_c", &err);
	checkOpenCLErrors(err, "Failed to create kernel FWD_init_alpha")

	kernel2 = clCreateKernel(prog, "kmeans_swap", &err);
	checkOpenCLErrors(err, "Failed to create kernel FWD_norm_alpha")

	clReleaseProgram(prog);	

	// use SVM to create buffer
	
	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1;}
	d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", n_points * n_features, err); return -1;}
	d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", n_clusters * n_features, err); return -1;}
	d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", n_points, err); return -1;}
		
	// cpu write to the buffer using map and unmap 
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1; }
	
	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);
	
	size_t global_work[3] = { (size_t)n_points, 1, 1 };
	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size= BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51

	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	
	membership_OCL = (int*) malloc(n_points * sizeof(int));
}


float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{    
    int      i, j, n = 0;		/* counters */
    int		 loop=0, temp;
    int     *new_centers_len;	/* [nclusters]: no. of points in each cluster */
    float    delta;				/* if the point moved */
    float  **clusters;			/* out: [nclusters][nfeatures] */
    float  **new_centers;		/* [nclusters][nfeatures] */

	int     *initial;			/* used to hold the index of points not yet selected
								   prevents the "birthday problem" of dual selection (?)
								   considered holding initial cluster indices, but changed due to
								   possible, though unlikely, infinite loops */
	int      initial_points;
	int		 c = 0;

	/* nclusters should never be > npoints
	   that would guarantee a cluster without points */
	if (nclusters > npoints)
		nclusters = npoints;

	// fixme : use svm 
    /* allocate space for and initialize returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

	/* initialize the random clusters */
	initial = (int *) malloc (npoints * sizeof(int));
	for (i = 0; i < npoints; i++)
	{
		initial[i] = i;
	}
	initial_points = npoints;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters && initial_points >= 0; i++) {
		//n = (int)rand() % initial_points;		
		
		// fixme: use svm
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[initial[n]][j];	// remapped

		/* swap the selected index to the end (not really necessary,
		   could just move the end up) */
		temp = initial[n];
		initial[n] = initial[initial_points-1];
		initial[initial_points-1] = temp;
		initial_points--;
		n++;
    }

	/* initialize the membership to -1 for all */
	// fixme: use svm
    for (i=0; i < npoints; i++)
	  membership[i] = -1;

    /* allocate space for and initialize new_centers_len and new_centers */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

	/* iterate until convergence */
	do {
        delta = 0.0;
		// CUDA
		delta = (float) kmeansOCL(feature,			/* in: [npoints][nfeatures] */
								   nfeatures,		/* number of attributes for each point */
								   npoints,			/* number of data points */
								   nclusters,		/* number of clusters */
								   membership,		/* which cluster the point belongs to */
								   clusters,		/* out: [nclusters][nfeatures] */
								   new_centers_len,	/* out: number of points in each cluster */
								   new_centers		/* sum of points in each cluster */
								   );

		/* replace old cluster centers with new_centers */
		/* CPU side of reduction */
		for (i=0; i<nclusters; i++) {
			for (j=0; j<nfeatures; j++) {
				if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
				new_centers[i][j] = 0.0;	/* set back to 0 */
			}
			new_centers_len[i] = 0;			/* set back to 0 */
		}	 
		c++;
    } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */
	printf("iterated %d times\n", c);
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

int	kmeansOCL(float **feature,    // in: [npoints][nfeatures]
           int     n_features,
           int     n_points,
           int     n_clusters,
           int    *membership,
		   float **clusters,
		   int     *new_centers_len,
           float  **new_centers)	
{
  
	int delta = 0;
	int i, j, k;
	cl_int err = 0;
	
	size_t global_work[3] = { (size_t)n_points, 1, 1 }; 

	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size=BLOCK_SIZE2; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;
	
	// fixme: use svm
	err = clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err); return -1; }

	int size = 0; int offset = 0;
					
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);
	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	clFinish(cmd_queue);

	// fixme : use svm
	err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
	
	delta = 0;
	for (i = 0; i < n_points; i++)
	{
		int cluster_id = membership_OCL[i];
		new_centers_len[cluster_id]++;
		if (membership_OCL[i] != membership[i])
		{
			delta++;
			membership[i] = membership_OCL[i];
		}
		for (j = 0; j < n_features; j++)
		{
			new_centers[cluster_id][j] += feature[i][j];
		}
	}

	return delta;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
		float *pt2,
		int    numdims)
{
	int i;
	float ans=0.0;

	for (i=0; i<numdims; i++)
		ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

	return(ans);
}

/*----< find_nearest_point() >-----------------------------------------------*/
__inline
int find_nearest_point(float  *pt,          /* [nfeatures] */
		int     nfeatures,
		float  **pts,         /* [npts][nfeatures] */
		int     npts)
{
	int index, i;
	float max_dist=FLT_MAX;

	/* find the cluster center id with min distance to pt */
	for (i=0; i<npts; i++) {
		float dist;
		dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
		if (dist < max_dist) {
			max_dist = dist;
			index    = i;
		}
	}
	return(index);
}

/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/
float rms_err	(float **feature,         /* [npoints][nfeatures] */
		int     nfeatures,
		int     npoints,
		float **cluster_centres, /* [nclusters][nfeatures] */
		int     nclusters)
{
	int    i;
	int   nearest_cluster_index;	/* cluster center id with min distance to pt */
	float  sum_euclid = 0.0;		/* sum of Euclidean distance squares */
	float  ret;						/* return value */

	/* calculate and sum the sqaure of euclidean distance*/	
#pragma omp parallel for \
	shared(feature,cluster_centres) \
	firstprivate(npoints,nfeatures,nclusters) \
	private(i, nearest_cluster_index) \
	schedule (static)	
	for (i=0; i<npoints; i++) {
		nearest_cluster_index = find_nearest_point(feature[i], 
				nfeatures, 
				cluster_centres, 
				nclusters);

		sum_euclid += euclid_dist_2(feature[i],
				cluster_centres[nearest_cluster_index],
				nfeatures);

	}	
	/* divide by n, then take sqrt */
	ret = sqrt(sum_euclid / npoints);

	return(ret);
}

void deallocateMemory()
{
	clReleaseMemObject(d_feature);
	clReleaseMemObject(d_feature_swap);
	clReleaseMemObject(d_cluster);
	clReleaseMemObject(d_membership);
	free(membership_OCL);
}

int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device ) clReleaseDevice( device );

	// reset all variables
	cmd_queue = 0;
	context = 0;
	//device_list = 0;
	//num_devices = 0;
	//device_type = 0;

	return 0;
}
