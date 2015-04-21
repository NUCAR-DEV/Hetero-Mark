#ifndef KMEANS_CL12_KMEANS_H
#define KMEANS_CL12_KMEANS_H

#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "../common/Benchmark.h"

/**
 * KMeans benchmark
 */
class KMeans : public Benchmark 
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
	
	// Kernel Launcher
	std::unique_ptr<KernelLauncher> iir_kernel;

	// Init kernels
	void InitKernels();

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

	/**
	 * Summarize
	 */
	void Summarize() override;

	/**
	 * Set data length
	 */
	void setDataLength(uint32_t length) { len = length; }
};

#endif
