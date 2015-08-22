include(CppLint.cmake)
set( FOLDER_NAME Hetero-Mark )
set( SUBDIRECTORIES src )

set( CMAKE_CXX_FLAGS "-Wall -Werror" )
ADD_DEFINITIONS(-std=c++11)

# Include from path root
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Style check
set( check_target check )
set( check_target_src )
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")

# Test
enable_testing()
add_subdirectory(lib/gtest-1.7.0)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

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

