/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/******************************************************************************/
/* Modified by Leiming Yu (ylm@ece.neu.edu)                                   */
/*             Northeastern University                                        */
/******************************************************************************/

#ifndef _H_FUZZY_KMEANS
#define _H_FUZZY_KMEANS


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <float.h>

#include "src/common/cl_util/cl_util.h"
#include "src/common/benchmark/benchmark.h"


#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
#endif


#ifndef FLT_MAX
  #define FLT_MAX 3.40282347e+38
#endif


#ifdef NV                                                                       
    #include <oclUtils.h>                                                       
#else                                                                           
	#include <CL/cl.h>                                                          
#endif  

#define _CRT_SECURE_NO_DEPRECATE 1
#define RANDOM_MAX 2147483647


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


extern double wtime(void);

using namespace clHelper;

struct FilePackage
        {
        	char* filename;
        	int binary;
        	double threshold;
        	int max_cl;
        	int min_cl;
        	int RMSE;
        	int output;
        	int nloops;
        };

class KMEANS : public Benchmark
{
public:
	KMEANS();
	~KMEANS();
	void Initialize() override {};
	void Run() override;
	void Verify() override {}
	void Cleanup() override { CleanUpKernels(); }
	void Summarize() override {}
	void SetInitialParameters(FilePackage parameters);

private:
	//-----------------------------------------------------------------------//
	// Host Parameters
	//-----------------------------------------------------------------------//

	// command line options 
	char   *filename;                                                       
	int isBinaryFile;                                                       
	int isOutput;                                            
	int npoints;                                                            
	int nfeatures;                                                          
	int max_nclusters;           
	int min_nclusters;          
	int isRMSE;                                                             
	int nloops;                 // number of iterations for each cluster            
	int   index;                // number of iteration to reach the best RMSE
	float threshold;            

	float **feature;           // host feature                                                


	//-----------------  Clustering parameters ------------------------------//
	int nclusters;               // number of clusters
	int *membership;			// which cluster a data point belongs to
	// hold coordinates of cluster centers
	float **tmp_cluster_centres; // pointer to the clusters
	float **cluster_centres;     // pointer to the clusters

	//-----------------  Create_mem -----------------------------------------//
	int   *membership_OCL;

	//----------------- Kmeans_clustering parameters ------------------------//
	float  **clusters;          // out: [nclusters][nfeatures]                  
	int     *initial;           // used to hold the index of points not yet selected
	int     *new_centers_len;   // [nclusters]: no. of points in each cluster   
	float  **new_centers;       // [nclusters][nfeatures]               
	float    delta;             // if the point moved                           

	//----------------- rms_err parameters ----------------------------------//
	float rmse;				    // RMSE for each clustering
	float min_rmse;			
	float	min_rmse_ref;		
	int	best_nclusters;

	//-----------------------------------------------------------------------//
	// Device Parameters
	//-----------------------------------------------------------------------//
	// Helper objects
	clRuntime *runtime;
	clFile *file;

	// ocl kernel
	cl_kernel kernel_s;
	cl_kernel kernel2;
	
	bool svmCoarseGrainAvail;
	bool svmFineGrainAvail;

	// ocl resources
	cl_platform_id   platform;
	cl_context	     context;
	cl_device_id     device;
	cl_command_queue cmd_queue;
	cl_program       prog;
	
	// device memory
	cl_mem d_feature;          // device feature
	cl_mem d_feature_swap;
	cl_mem d_cluster;          // cluster
	cl_mem d_membership;

	//-----------------------------------------------------------------------//
	// I/O function
	//-----------------------------------------------------------------------//

	//-----------------------------------------------------------------------//
	// Cluster function
	//-----------------------------------------------------------------------//
	void Clustering();
	void CL_initialize();
	void CL_build_program();
	void CL_create_kernels();
	void Create_mem();
	void Swap_features();
	void Kmeans_clustering();
	void Kmeans_ocl();

	//-----------------------------------------------------------------------//
	// rms function
	//-----------------------------------------------------------------------//
	float euclid_dist_2(float*, float*); 
	int find_nearest_point(float*, float**);
	void RMS_err();

	//-----------------------------------------------------------------------//
	// Command line ouput  
	//-----------------------------------------------------------------------//
	void Display_results();

	//-----------------------------------------------------------------------//
	// Clean functions 
	//-----------------------------------------------------------------------//
	void CleanUpKernels();
	void Free_mem();

};


#endif
