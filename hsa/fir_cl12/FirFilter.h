#ifndef FIR_CL12_FIRFILTER_H
#define FIR_CL12_FIRFILTER_H

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "../common/Benchmark.h"

/**
 * FirFilter benchmark
 */
class FirFilter : public Benchmark 
{

	// Dataset
	uint32_t len = 1024;
	float *coeff;
	float *in;
	float *out;
	uint32_t numTap;
	
	
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
		void *output;
		void *coeff;
		void *input;
		uint32_t numTap;
	} args;

	// Init params
	void InitParam();

public:

	/**
	 * Constructor
	 */
	FirFilter();

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

	/**
	 * Set data length
	 */
	void setDataLength(uint32_t length) { len = length; }
};

#endif
