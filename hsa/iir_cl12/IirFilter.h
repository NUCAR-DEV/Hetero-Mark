#include "../common/HsaHelper.h"
#include "../common/Benchmark.h"

class IirFilter : public Benchmark 
{
	
	// Hsa helper
	HsaHelper helper;

public:

	/**
	 * Constructor
	 */
	IirFilter();

	/**
	 * Run
	 */
	void Run() override;
};
