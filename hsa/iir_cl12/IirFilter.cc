#include "../common/HsaKernelLauncher.h"
#include "../common/HsaHelper.h"

#include "IirFilter.h"

IirFilter::IirFilter() :
	helper(),
	kernel_launcher()
{
}


void IirFilter::Init() 
{
	// Init helper
	helper.Init();

	// Load kernels
	helper.LoadProgram("kernel.brig");

	// Setup kernel launcher
	kernel_launcher.setHelper(&helper);
	kernel_launcher.setName("&__OpenCL_ParIIR_kernel");
	kernel_launcher.Init();
}


void IirFilter::Run()
{
}


void IirFilter::Verify()
{
}


