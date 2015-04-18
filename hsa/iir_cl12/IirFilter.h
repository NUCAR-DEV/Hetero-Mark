#ifndef IIR_CL12_IIRFILTER_H
#define IIR_CL12_IIRFILTER_H

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "../common/Benchmark.h"

/**
 * IirFilter benchmark
 */
class IirFilter : public Benchmark 
{

	// Dataset
	uint32_t len = 1024;
	uint32_t channels;
	uint32_t rows;
	float c;
	float *nsec;
	float *dsec;
	float *in;
	float *out;
	
	
	// Hsa helper
	HsaHelper helper;

	// Kernel Launcher
	HsaKernelLauncher kernel_launcher;

	// Attribute
	struct __attribute__ ((aligned(16))) args_t 
	{
		uint64_t global_offset_0;
		uint64_t global_offset_1;
		uint64_t global_offset_2;
		uint64_t printf_buffer;
		uint64_t vqueue_pointer;
		uint64_t aqlwrap_pointer;
		uint32_t len;
		float c;
		void *nsec;
		void *dsec;
		void *sm;
		void *x;
		void *y;
	} args;

	// Init params
	void InitParam();

public:

	/**
	 * Constructor
	 */
	IirFilter();

	/**
	 * Init
	 */
	void Init() override;

	/**
	 * Run
	 */
	void Run() override;

	/**
	 * Verify
	 */
	void Verify() override;
};

#endif
