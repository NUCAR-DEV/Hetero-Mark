#include <cstdio>
#include <cstdlib>

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
