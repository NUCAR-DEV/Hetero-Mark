#include "hsa.h"
#include "hsa_ext_finalize.h"

/**
 * An HsaHelper is an HSA runtime environment and a wraper of HSA runtime.
 */
class HsaHelper 
{
	// The gpu that all the following task will be dispatched on
	hsa_agent_t gpu;

	// Check erro
	void CheckError(hsa_status_t err, const char *information);

	// Call back function used when iterating devices to find a GPU device
	static hsa_status_t FindGpuDevice(hsa_agent_t agent, void *data);

	/**
	 * Get GPU device
	 */
	hsa_agent_t getDevice();

	/**
	 * Create queue
	 */
	hsa_queue_t *CreateQueue()

public:
	
	/**
	 * Init Hsa Environment
	 */
	void Init();

};
