# Check if snack is available
find_file(have_snack NAMES snack.sh DOC "snack.sh file") 

# Check if HSA runtime is available
find_library(HSA_RUNTIME 
  NAMES hsa-runtime64
  PATHS $ENV{HSA_RUNTIME_PATH}/lib
  NO_DEFAULT_PATH)

if (have_snack AND HSA_RUNTIME)
else (have_snack AND HSA_RUNTIME)
  message("snack.sh or hsa runtime not found, skipping HSA benchmarks")
endif ()
