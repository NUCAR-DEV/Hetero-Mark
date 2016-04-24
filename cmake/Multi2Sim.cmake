option(CompileForMulti2Sim "Compile in 32-bit mode and statically link with Multi2Sim HSA runtime library." OFF)

if (CompileForMulti2Sim)
  set (M2S_ROOT $ENV{M2S_ROOT})
  if (DEFINED M2S_ROOT)
  else ()
    message(FATAL_ERROR "Environment variable M2S_ROOT is required to compile for Multi2Sim")
  endif()

  if (have_snack)
  else()
    message(FATAL_ERROR "SNACK is required for compiling for Mutl2Sim")
  endif()

  set (HSA_RUNTIME ${M2S_ROOT}/lib/.libs/libm2s-hsa.a)
  message(STATUS ${HSA_RUNTIME})

  if (HSA_RUNTIME)
  else ()
    message(FATAL_ERROR "Multi2Sim HSA runtime library not found")
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
  set(SNACK ${SNACK} -m32)

endif (CompileForMulti2Sim)
