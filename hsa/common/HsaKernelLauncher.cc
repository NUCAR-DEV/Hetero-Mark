#include <cstring>

#include "HsaHelper.h"
#include "HsaKernelLauncher.h"

hsa_status_t HsaKernelLauncher::FindKernargMemoryRegion(hsa_region_t region, 
		void *data)
{
	hsa_region_segment_t segment;
	hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
	if (HSA_REGION_SEGMENT_GLOBAL != segment)
	{
		return HSA_STATUS_SUCCESS;
	}

	hsa_region_global_flag_t flags;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
	{
		hsa_region_t *ret = (hsa_region_t *)data;
		*ret = region;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}


void HsaKernelLauncher::Init()
{
	hsa_status_t err;

	// Retrieve symbol
	hsa_executable_symbol_t symbol;
	err = hsa_executable_get_symbol(helper->getExecutable(), 
			"", name.c_str(), helper->getGpu(), 0, &symbol);
	helper->CheckError(err, "Get symbol");

	// Get kernel object
	err = hsa_executable_symbol_get_info(symbol, 
			HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, 
			&kernel_object);
	helper->CheckError(err, "Get kernel object");
	err = hsa_executable_symbol_get_info(symbol,
			HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
			&kernarg_segment_size);
	helper->CheckError(err, "Get kernel argument segment size");
}


void HsaKernelLauncher::PrepareArgument()
{
	hsa_status_t err;

	// Find a memory region that supports kernel arguments
	hsa_region_t kernarg_region;
	kernarg_region.handle = (uint64_t)-1;
	hsa_agent_iterate_regions(helper->getGpu(), 
			HsaKernelLauncher::FindKernargMemoryRegion,
			&kernarg_region);
	err = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR 
		: HSA_STATUS_SUCCESS;
	helper->CheckError(err, "Find kernarg memory region");

	// Kernel argument buffer
	err = hsa_memory_allocate(kernarg_region, kernarg_segment_size, 
			&kernarg_address);
	helper->CheckError(err, "Allocate memory buffer");
	memcpy(kernarg_address, arguments, kernarg_segment_size);

}


void HsaKernelLauncher::LaunchKernel()
{
	hsa_status_t err;

	// Prepare argument
	PrepareArgument();

	// Create a signal to wait for dispatch to finish
	hsa_signal_t signal;
	err = hsa_signal_create(1, 0, NULL, &signal);
	helper->CheckError(err, "Create signal");

	// Obtail the current queue write index
	uint64_t index = hsa_queue_load_write_index_relaxed(helper->getQueue());

	// Write the aql packet 
	const uint32_t queueMask = helper->getQueue()->size - 1;
	hsa_kernel_dispatch_packet_t *dispatch_packet = 
		&(((hsa_kernel_dispatch_packet_t *)
		(helper->getQueue()->base_address))[index & queueMask]);
	dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << 
		HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
	dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << 
		HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
	dispatch_packet->setup  |= 1 << 
		HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
	dispatch_packet->workgroup_size_x = (uint16_t)group_size[0];
	dispatch_packet->workgroup_size_y = (uint16_t)group_size[1];
	dispatch_packet->workgroup_size_z = (uint16_t)group_size[2];
	dispatch_packet->grid_size_x = global_size[0];
	dispatch_packet->grid_size_y = global_size[1];
	dispatch_packet->grid_size_z = global_size[2];
	dispatch_packet->completion_signal = signal;
	dispatch_packet->kernel_object = kernel_object;
	dispatch_packet->kernarg_address = (void*) kernarg_address;
	dispatch_packet->private_segment_size = private_segment_size;
	dispatch_packet->group_segment_size = group_segment_size;
	__atomic_store_n((uint8_t*)(&dispatch_packet->header), 
			(uint8_t)HSA_PACKET_TYPE_KERNEL_DISPATCH, 
			__ATOMIC_RELEASE);

	// Increment the write index and ring the doorbell to dispatch the kernel
	hsa_queue_store_write_index_relaxed(helper->getQueue(), index + 1);
	hsa_signal_store_relaxed(helper->getQueue()->doorbell_signal, index);
	helper->CheckError(err, "Dispatching kernel");

	// Wait signal
	hsa_signal_value_t value = hsa_signal_wait_acquire(signal, 
			HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
			HSA_WAIT_STATE_BLOCKED);

}
