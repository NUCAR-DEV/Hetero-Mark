if (COMPILE_CUDA)
	find_package(CUDA)

	set(CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS};
		-std=c++11 -gencode arch=compute_61,code=sm_61)

	message(${CUDA_FOUND})
	message(${CUDA_VERSION})

	message("\
-- compile cuda benchmark. \
")
else()
	message("-- nvcc not used, will skip compiling cuda benchmarks")
endif()



