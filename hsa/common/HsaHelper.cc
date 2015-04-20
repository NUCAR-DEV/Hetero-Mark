#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "HsaHelper.h"

void HsaHelper::CheckError(hsa_status_t err, const char *information) 
{
	if (err) 
	{
		printf("Error(%d): %s\n", err, information);
		exit(err);
	}
	else
	{
		if (is_verification_mode)
			printf("Succeed: %s\n", information);
	}
}


hsa_status_t HsaHelper::FindGpuDevice(hsa_agent_t agent, void *data)
{
	hsa_status_t status;
	hsa_device_type_t device_type;
	status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
	if (status == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU)
	{
		hsa_agent_t *ret = (hsa_agent_t *)data;
		*ret = agent;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}


void HsaHelper::Init() 
{
	// Init hsa runtime
	hsa_status_t err = hsa_init();
	CheckError(err, "Init HSA runtime");

	// Get GPU agent
	gpu = getDevice();

	// Get queue
	queue = CreateQueue();

	// Get the isa of the gpu
	err = hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa);
	CheckError(err, "Get GPU ISA");
}


hsa_agent_t HsaHelper::getDevice()
{
	hsa_agent_t agent;
	hsa_status_t err;
	err = hsa_iterate_agents(HsaHelper::FindGpuDevice, &agent);
	if (err == HSA_STATUS_INFO_BREAK) 
		err = HSA_STATUS_SUCCESS;
	CheckError(err, "Find GPU device");
	return agent;
}


hsa_queue_t *HsaHelper::CreateQueue()
{
	// Check queue size
	uint32_t queue_size = 0;
	hsa_status_t err = hsa_init();
	err = hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, 
			&queue_size);
	CheckError(err, "Query maximum queue size");

	// Create queue
	hsa_queue_t *queue;
	err = hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_SINGLE,
			NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
	CheckError(err, "Create queue");

	// Return queue
	return queue;
}


void HsaHelper::LoadProgram(const char *file)
{
	hsa_status_t err;

	// Load module
	hsa_ext_module_t module;
	LoadModuleFromFile(file, &module);

	// Create program
	hsa_ext_program_t program;
	memset(&program, 0, sizeof(hsa_ext_program_t));
	err = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, 
			HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, 
			&program);
	CheckError(err, "Create program");

	// Add module
	err = hsa_ext_program_add_module(program, module);
	CheckError(err, "Add module to program");

	// Finalize
	hsa_code_object_t code_object;
	Finalize(program, &code_object);

	// Destory program
	err = hsa_ext_program_destroy(program);
	CheckError(err, "Destroy program");
	
	// CreateExecutable
	CreateExecutable(code_object);
}


void HsaHelper::LoadModuleFromFile(const char *file_name,
		hsa_ext_module_t *module)
{
	FILE *fp = fopen(file_name, "rb");
	if (fp == NULL) 
	{
		printf("Fail to open file %s.\n", file_name);
		exit(1);
	}
	fseek(fp, 0, SEEK_END);
	size_t file_size = (size_t) (ftell(fp) * sizeof(char));
	fseek(fp, 0, SEEK_SET);
	char *buf = (char *)malloc(file_size);
	memset(buf, 0, file_size);
	size_t read_size = fread(buf, sizeof(char), file_size, fp);
	if(read_size != file_size) 
	{
		free(buf);
	}
	else
	{
		*module = (hsa_ext_module_t)buf;
	}
	fclose(fp);
}


void HsaHelper::Finalize(hsa_ext_program_t program, 
		hsa_code_object_t *code_object)
{
	// Create control directive
	hsa_ext_control_directives_t control_directives;
	memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

	// Finalize code
	hsa_status_t err;
	err = hsa_ext_program_finalize(program, isa, 0, control_directives, "",
			HSA_CODE_OBJECT_TYPE_PROGRAM, code_object);
	CheckError(err, "Finalize program");
}


void HsaHelper::CreateExecutable(hsa_code_object_t code_object)
{
	hsa_status_t err;

	// Create executable
	err = hsa_executable_create(HSA_PROFILE_FULL, 
			HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
	CheckError(err, "Create executable");

	// Load code object
	err = hsa_executable_load_code_object(executable, gpu, code_object, "");
	CheckError(err, "Executable load code object");

	// Freeze the executable
	err = hsa_executable_freeze(executable, "");
	CheckError(err, "Freeze the executable");
}


void HsaHelper::RegisterMemory(void *pointer, size_t size)
{
	hsa_status_t err;
	err = hsa_memory_register(pointer, size);
	CheckError(err, "Register HSA memory");
}
