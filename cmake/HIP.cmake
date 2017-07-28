if (COMPILE_HIP)
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /opt/rocm/hip/cmake)
	include(FindHIP)

	if (HIP_FOUND)
		message("-- hipcc found, will compile hip benchmarks.")
		include_directories("/opt/rocm/hip/include")
		set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} -std=c++11)
	else (HIP_FOUND)
		message("--hip not found")
		set(COMPILE_HIP Off)
	endif(HIP_FOUND)

endif (COMPILE_HIP)
