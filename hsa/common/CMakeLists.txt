cmake_minimum_required (VERSION 2.6)
set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")

add_library(HsaHelper HsaHelper.cc)

include_directories(${HSA_RUNTIME_PATH}/include)

