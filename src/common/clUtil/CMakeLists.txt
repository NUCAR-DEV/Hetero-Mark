set( FOLDER_NAME clUtil )
set( HEADER_FILE 
  clError.h
  clFile.h
  clProfiler.h
  clRuntime.h
  clUtil.h
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${HEADER_FILE})
add_style_check_target(${check_target} "${check_target_src}" "${SUBDIRECTORIES}")



###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )
