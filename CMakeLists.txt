include(ExternalProject)
include(CppLint.cmake)
set( FOLDER_NAME Hetero-Mark )
set( SUBDIRECTORIES src )

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

set( CMAKE_CXX_FLAGS "-Wall -Werror -std=c++11 -pthread" )

option (hetero_mark_test "Compile all Hetero-mark unit tests" OFF)

# Include from path root
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Style check
set( check_target check )
set( check_target_src )
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Test
if (hetero_mark_test)
  enable_testing()
  ExternalProject_Add(googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(googletest binary_dir)
  ExternalProject_Get_Property(googletest source_dir)
  add_library(gtest      UNKNOWN IMPORTED)
  add_library(gtest_main UNKNOWN IMPORTED)
  set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libgtest.a)
  set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libgtest_main.a)
  add_dependencies(gtest      googletest)
  add_dependencies(gtest_main googletest)
  include_directories(${source_dir}/include)
endif (hetero_mark_test)

############################################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property( GLOBAL PROPERTY USE_FOLDERS ON )
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

foreach( subdir ${SUBDIRECTORIES} )
  add_subdirectory( ${subdir} )
endforeach( subdir )

