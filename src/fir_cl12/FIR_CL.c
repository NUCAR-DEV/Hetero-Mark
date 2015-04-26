
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "eventlist.h"
#include <CL/cl.h>
//#ifdef GPUPROF
//#include "inc/GPUPerfAPI.h"
//#include <dlfcn.h>
//#endif


#define CHECK_STATUS( status, message )   \
		if(status != CL_SUCCESS) \
		{ \
			printf( message); \
			printf( "\n" ); \
			return 1; \
		}



//#ifdef GPUPROF
//char* createFilenameWithTimestamp();
//void WriteSession( gpa_uint32 currentWaitSessionID, char* filename );
//#endif

/** Define custom constants*/
#define MAX_SOURCE_SIZE (0x100000)

cl_uint numTap = 0;
cl_uint numData = 0;		// Block size
cl_uint numTotalData = 0;
cl_uint numBlocks = 0;
cl_float* input = NULL;
cl_float* output = NULL;
cl_float* coeff = NULL;
cl_float* temp_output = NULL;


int main(int argc , char** argv) {
  double start = time_stamp();
	/** Define Custom Variables */
	int i,count;
	int local;

	if (argc < 3)
	{
		printf(" Usage : ./auto_exec.sh <numBlocks> <numData>\n");
		exit(0);
	}
	if (argc > 1)
	{
		numBlocks = atoi(argv[1]);
		numData = atoi(argv[2]);
	}


	/** Declare the Filter Properties */
	numTap = 1024;
	numTotalData = numData * numBlocks;
	local = 64;

	printf("FIR Filter\n Data Samples : %d \n NumBlocks : %d \n Local Workgroups : %d\n", numData,numBlocks,local);
	//exit(0);
	/** Define variables here */
	input = (cl_float *) malloc( numTotalData* sizeof(cl_float) );
	output = (cl_float *) malloc( numTotalData* sizeof(cl_float) );
	coeff = (cl_float *) malloc( numTap* sizeof(cl_float) );
	temp_output = (cl_float *) malloc( (numData+numTap-1) * sizeof(cl_float) );

	/** Initialize the input data */
	for( i=0;i<numTotalData;i++ )
	{
		input[i] = 8;
		output[i] = 99;
	}

	for( i=0;i<numTap;i++ )
		coeff[i] = 1.0/numTap;

	for( i=0; i<(numData+numTap-1); i++ )
		temp_output[i] = 0.0;



	// Event Creation
	cl_event event;
	EventList* eventList;

#if 1
	// Read the input file
	FILE *fip;
	i=0;
	fip = fopen("temp.dat","r");
	while(i<numTotalData)
	{
		fscanf(fip,"%f",&input[i]);
		i++;
	}
	fclose(fip);

#if 0
	printf("\n The Input:\n");
	i = 0;
	while( i<numTotalData )
	{
		printf( "%f, ", input[i] );

		i++;
	}
#endif
#endif

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("FIR.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1,
			&device_id, &ret_num_devices);

	printf("/n No of Devices %d",ret_num_platforms );

	char *platformVendor;
	size_t platInfoSize;
	clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL,
			&platInfoSize);

	platformVendor = (char*)malloc(platInfoSize);

	clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, platInfoSize,
			platformVendor, NULL);
	printf("\tVendor: %s\n", platformVendor);
	free(platformVendor);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

	
	// Create Eventlist for Timestamps
	eventList = new EventList(context, command_queue, device_id,true);

//#ifdef GPUPROF
//	// Create performance counter Init
//	GPA_Initialize();
//	GPA_OpenContext( command_queue );
//	GPA_EnableAllCounters();
//#endif

	// Create memory buffers on the device for each vector
	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(cl_float) * numData, NULL, &ret);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * numData, NULL, &ret);
	cl_mem coeffBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(cl_float) * numTap, NULL, &ret);
	cl_mem temp_outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float) * (numData+numTap-1), NULL, &ret);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	CHECK_STATUS( ret,"Error: Build Program\n");

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "FIR", &ret);
	CHECK_STATUS( ret,"Error: Create kernel. (clCreateKernel)\n");


	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&coeffBuffer);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&temp_outputBuffer);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&numTap);

	// Initialize Memory Buffer
	ret = clEnqueueWriteBuffer(command_queue,
			coeffBuffer,
			1,
			0,
			numTap * sizeof(cl_float),
			coeff,
			0,
			0,
			&event);

	eventList->add(event);

	ret = clEnqueueWriteBuffer(command_queue,
			temp_outputBuffer,
			1,
			0,
			(numTap) *sizeof(cl_float),
			temp_output,
			0,
			0,
			&event);

	eventList->add(event);



	// Decide the local group formation
	size_t globalThreads[1]={numData};
	size_t localThreads[1]={128};
	cl_command_type cmdType;
	count = 0;
	while( count < numBlocks )
	{

		/* fill in the temp_input buffer object */
		ret = clEnqueueWriteBuffer(command_queue,
				temp_outputBuffer,
				1,
				(numTap-1)*sizeof(cl_float),
				numData * sizeof(cl_float),
				input + (count * numData),
				0,
				0,
				&event);

		//(numTap-1)*sizeof(cl_float)
		eventList->add(event);


		size_t global_item_size = numData; // GLOBAL ITEMSIZE IS CUSTOM BASED ON COMPUTAION ALGO
		size_t local_item_size = numData;
		//		size_t local_item_size[4] = {numData/4,numData/4,numData/4,numData/4}; // LOCAL ITEM SIZE IS CUSTOM BASED ON COMPUTATION ALGO

//#if GPUPROF
//
//
//		//Session Creation
//		static gpa_uint32 currentWaitSessionID = 1;
//		gpa_uint32 sessionID;
//		GPA_BeginSession( &sessionID );
//
//		// Required number of passes can be custom
//		gpa_uint32 numRequiredPasses = 1;
//
//
//		GPA_GetPassCount( &numRequiredPasses );
//		for ( gpa_uint32 i = 0; i < numRequiredPasses; i++ )
//		{
//			GPA_BeginPass();
//			GPA_BeginSample( 0 );
//
//#endif
//
			// Execute the OpenCL kernel on the list
			ret = clEnqueueNDRangeKernel(
					command_queue,
					kernel,
					1,
					NULL,
					globalThreads,
					localThreads,
					0,
					NULL,
					&event);


			CHECK_STATUS( ret,"Error: Range kernel. (clCreateKernel)\n");
			ret = clWaitForEvents(1, &event);
			ret = clWaitForEvents(1, &event);




//#if GPUPROF
//			// End Profile session
//			GPA_EndSample();
//			GPA_EndPass();
//
//			eventList->add(event);
//		}
//		GPA_EndSession();
//
//		bool readyResult = true;
//		if ( sessionID != currentWaitSessionID )
//		{
//			GPA_Status sessionStatus;
//			sessionStatus = GPA_IsSessionReady( &readyResult,
//					currentWaitSessionID );
//			while ( sessionStatus == GPA_STATUS_ERROR_SESSION_NOT_FOUND )
//			{
//				// skipping a session which got overwritten
//				currentWaitSessionID++;
//				sessionStatus = GPA_IsSessionReady( &readyResult,
//						currentWaitSessionID );
//			}
//		}
//		if ( readyResult )
//		{
//			WriteSession( currentWaitSessionID,
//					"Results1.log" );
//			currentWaitSessionID++;
//		}
//		//Cleanup
//		GPA_CloseContext();
//
//#endif

		eventList->add(event);

		/* Get the output buffer */
		ret = clEnqueueReadBuffer(
				command_queue,
				outputBuffer,
				CL_TRUE,
				0,
				numData * sizeof( cl_float ),
				output + count * numData,
				0,
				NULL,
				&event );
		eventList->add(event);
		count ++;
	}

	/* Uncomment to trint output */
	//printf("\n The Output:\n");
	//i = 0;
	//while( i<numTotalData )
	//{
	//	printf( "%f ", output[i] );

	//	i++;
	//}

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(inputBuffer);
	ret = clReleaseMemObject(outputBuffer);
	ret = clReleaseMemObject(coeffBuffer);
	ret = clReleaseMemObject(temp_outputBuffer);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(input);
	free(output);
	free(coeff);
	free(temp_output);

//#ifdef GPUPROF
//	GPA_Destroy();
//#endif

	/* comment to hide timing events */
	eventList->printEvents();
	eventList->dumpEvents("eventDumps");
	delete eventList;
	return 0;

	double start = time_stamp();
	printf("Total time = %f ms\n", end - start);
}

//#ifdef GPUPROF
//// Perfoemance API write function
//void WriteSession( gpa_uint32 currentWaitSessionID,  char* filename )
//{
//	static bool doneHeadings = false;
//	gpa_uint32 count;
//	GPA_GetEnabledCount( &count );
//	FILE* f;
//	if ( !doneHeadings )
//	{
//		const char* name;
//		f = fopen( filename, "w+" );
//		assert( f );
//		fprintf( f, "Identifier, " );
//		for ( gpa_uint32 counter = 0 ; counter < count ; counter++ )
//		{
//			gpa_uint32 enabledCounterIndex;
//			GPA_GetEnabledIndex( counter, &enabledCounterIndex );
//			GPA_GetCounterName( enabledCounterIndex, &name );
//			fprintf( f, "%s, ", name );
//		}
//		fprintf( f, "\n" );
//		fclose( f );
//		doneHeadings = true;
//	}
//	f = fopen( filename, "a+" );
//	assert( f );
//	gpa_uint32 sampleCount;
//	GPA_GetSampleCount( currentWaitSessionID, &sampleCount );
//	for ( gpa_uint32 sample = 0 ; sample < sampleCount ; sample++ )
//	{
//		fprintf( f, "session: %d; sample: %d, ", currentWaitSessionID,
//				sample );
//		for ( gpa_uint32 counter = 0 ; counter < count ; counter++ )
//		{
//			gpa_uint32 enabledCounterIndex;
//			GPA_GetEnabledIndex( counter, &enabledCounterIndex );
//			GPA_Type type;
//			GPA_GetCounterDataType( enabledCounterIndex, &type );
//			if ( type == GPA_TYPE_UINT32 )
//			{
//				gpa_uint32 value;
//				GPA_GetSampleUInt32( currentWaitSessionID,
//						sample, enabledCounterIndex, &value );
//				fprintf( f, "%u,", value );
//			}
//			else if ( type == GPA_TYPE_UINT64 )
//			{
//				gpa_uint64 value;
//				GPA_GetSampleUInt64( currentWaitSessionID,
//						sample, enabledCounterIndex, &value );
//				fprintf( f, "%I64u,", value );
//			}
//			else if ( type == GPA_TYPE_FLOAT32 )
//			{
//				gpa_float32 value;
//				GPA_GetSampleFloat32( currentWaitSessionID,
//						sample, enabledCounterIndex, &value );
//				fprintf( f, "%f,", value );
//			}
//			else if ( type == GPA_TYPE_FLOAT64 )
//			{
//				gpa_float64 value;
//				GPA_GetSampleFloat64( currentWaitSessionID,
//						sample, enabledCounterIndex, &value );
//				fprintf( f, "%f,", value );
//			}
//			else
//			{
//				assert(false);
//			}
//		}
//		fprintf( f, "\n" );
//	}
//	fclose( f );
//}
//#endif

