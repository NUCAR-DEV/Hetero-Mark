#ifndef COMMON_HSAHELPER_H
#define COMMON_HSAHELPER_H

#include <vector>
#include <string>

#include "hsa.h"
#include "hsa_ext_finalize.h"

/**
 * An HsaHelper is an HSA runtime environment and a wraper of HSA runtime.
 */
class HsaHelper 
{
	// The gpu that all the following task will be dispatched on
	hsa_agent_t gpu;

	// Isa of the gpu
	hsa_isa_t isa;

	// Queue
	hsa_queue_t *queue;

	// Executable
	hsa_executable_t executable;

	// Call back function used when iterating devices to find a GPU device
	static hsa_status_t FindGpuDevice(hsa_agent_t agent, void *data);

	// Get GPU device
	hsa_agent_t getDevice();

	//  Create queue
	hsa_queue_t *CreateQueue();

	// Load module from file
	void LoadModuleFromFile(const char *file, hsa_ext_module_t *module);	

	// Finalize
	void Finalize(hsa_ext_program_t program, hsa_code_object_t *code_object);

	// Create executable
	void CreateExecutable(hsa_code_object_t code_object);

public:
	/**
	 * Check error
	 */
	void CheckError(hsa_status_t err, const char *information);

	/**
	 * Init Hsa Environment
	 */
	void Init();

	/**
	 * Load kernels
	 */
	void LoadProgram(const char *file);

	/**
	 * Register memory
	 */
	void RegisterMemory(void *pointer, size_t size);

	/**
	 * Getters
	 */
	hsa_agent_t getGpu() const { return gpu; }
	hsa_executable_t getExecutable() const { return executable; }
	hsa_queue_t *getQueue() const { return queue; }
};

#endif
