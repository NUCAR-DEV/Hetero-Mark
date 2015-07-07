cmake_minimum_required (VERSION 2.6)

set (HSA_RUNTIME_PATH "$ENV{HSA_RUNTIME_PATH}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_GLAGS} -m32 -std=c++0x")

add_library(HsaHelper 
		HsaHelper.cc 
		HsaKernelLauncher.cc 
		OptionParser.cc 
		TimeKeeper.cc)

include_directories(${HSA_RUNTIME_PATH}/include)

