set(FOLDER_NAME command_line_option)
set(SUBDIRECTORIES)
set(SRC_FILES 
  benchmark_command_line_options.cc
  command_line_option.cc
  option_parser_impl.cc
  option_setting_help_printer.cc
  option_setting_impl.cc)
set( HEADER_FILES 
  argument.h
  argument_value.h
  argument_value_factory.h
  benchmark_command_line_options.h
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

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
if (COMPILE_HCC)
  set_target_properties(${FOLDER_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})
endif (COMPILE_HCC)
