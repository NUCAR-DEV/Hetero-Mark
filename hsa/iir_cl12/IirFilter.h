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
	
	// Hsa helper
	HsaHelper helper;

	// Kernel Launcher
	HsaKernelLauncher kernel_launcher;

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
