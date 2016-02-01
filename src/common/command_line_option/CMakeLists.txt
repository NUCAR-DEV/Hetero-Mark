set( FOLDER_NAME command_line_option )
set( SUBDIRECTORIES )
set( SRC_FILES 
  command_line_option.cc
  option_parser_impl.cc
  option_setting_help_printer.cc
  option_setting_impl.cc)
set( HEADER_FILES 
  argument.h
  argument_value.h
  argument_value_factory.h
  command_line_option.h
  option_parser.h
  option_parser_impl.h
  option_setting.h
  option_setting_help_printer.h
  option_setting_impl.h)
set( TEST_FILES
  option_parser_impl_test.cc
  option_setting_help_printer_test.cc
  option_setting_impl_test.cc)

# Unit test
if (enable-test)
  add_executable(${FOLDER_NAME}_test 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_test gtest gtest_main)
  add_test(${FOLDER_NAME}_test ${FOLDER_NAME}_test)
endif (enable-test)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
