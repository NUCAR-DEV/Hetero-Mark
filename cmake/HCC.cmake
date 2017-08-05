if (${CMAKE_CXX_COMPILER} MATCHES "hcc")
	message("\
-- hcc found, will compile hcc benchmarks. hcc will also be used to compile \
other benchmarks \
")

	set(COMPILE_HCC On)

	include_directories("/opt/rocm/include/hcc")
	include_directories("/opt/rocm/include/")

	# Thank for HCC-Example-Application for the following solution
	execute_process(COMMAND hcc-config  --cxxflags
		OUTPUT_VARIABLE KALMAR_COMPILE_FLAGS)
	set(COMPILER_FLAGS "${COMPILER_FLAGS} ${KALMAR_COMPILE_FLAGS}")
	message(${COMPILER_FLAGS})

	execute_process(COMMAND hcc-config  --ldflags
		OUTPUT_VARIABLE KALMAR_LINKER_FLAGS)
	set(LINKER_FLAGS "${LINKER_FLAGS} ${KALMAR_LINKER_FLAGS}")

else()
	message ("-- hcc not used, will skip compiling hc benchmarks")
endif ()
