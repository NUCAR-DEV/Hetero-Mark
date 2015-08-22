#set(CMAKE_DISABLE_SOURCE_CHANGES ON)
#set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( Hetero-Mark )

include(ExternalProject)
include(CppLint.cmake)
set( FOLDER_NAME Hetero-Mark )
set( SUBDIRECTORIES src )
set( CMAKE_CXX_FLAGS "-Wall -Werror -pthread" )
add_definitions(
  -std=c++11
)

option (enable-test "Compile all Hetero-mark unit tests" OFF)

# Include from path root
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

# Style check
set( check_target check )
set( check_target_src )
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Test
if (enable-test)
  enable_testing()
  ExternalProject_Add(googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(googletest binary_dir)
  ExternalProject_Get_Property(googletest source_dir)
  add_library(gtest UNKNOWN IMPORTED)
  add_library(gtest_main UNKNOWN IMPORTED)
  set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libgtest.a)
  set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libgtest_main.a)
  add_dependencies(gtest googletest)
  add_dependencies(gtest_main googletest)
  include_directories(${source_dir}/include)
endif (enable-test)


# Group samples by folder
set_property( GLOBAL PROPERTY USE_FOLDERS ON )
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

foreach( subdir ${SUBDIRECTORIES} )
  add_subdirectory( ${subdir} )
endforeach( subdir )

