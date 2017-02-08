option(COMPILE_HCC "Compile all the HC++ benchmarks and skip all other benchmarks")

if (COMPILE_HCC)
  # set(CMAKE_CXX_COMPILER hcc)
  message("Compiling HCC benchmarks, skipping others.")

  # Thank for HCC-Example-Application for the following solution
  execute_process(COMMAND hcc-config  --cxxflags 
                  OUTPUT_VARIABLE KALMAR_COMPILE_FLAGS)
  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${KALMAR_COMPILE_FLAGS}")
  
  execute_process(COMMAND hcc-config  --ldflags 
                  OUTPUT_VARIABLE KALMAR_LINKER_FLAGS)
  set(LINKER_FLAGS "${LINKER_FLAGS} ${KALMAR_LINKER_FLAGS}")

endif (COMPILE_HCC)
