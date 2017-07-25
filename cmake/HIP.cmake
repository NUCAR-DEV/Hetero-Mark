if (${CMAKE_CXX_COMPILER} MATCHES "hipcc")
	message("\
-- hipcc found, will compile hip benchmarks.
")

	set(COMPILE_HIP On)

else()
	message ("-- hipcc not used, will skip compiling hip benchmarks")
endif ()
