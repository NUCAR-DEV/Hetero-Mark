set(FOLDER_NAME memory)
# add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
# if (COMPILE_HCC)
#   set_target_properties(${FOLDER_NAME} PROPERTIES
#       COMPILE_FLAGS ${COMPILER_FLAGS}
#       LINK_FLAGS ${LINKER_FLAGS})
# endif (COMPILE_HCC)
set(SUBDIRECTORIES)
set(SRC_FILES 
)
set(HEADER_FILES 
  memory_manager.h
  array_view_memory_manager.h
)
set(TEST_FILES)


