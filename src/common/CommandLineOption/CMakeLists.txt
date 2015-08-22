set( FOLDER_NAME CommandLineOption )
set( SUBDIRECTORIES )
set( SRC_FILES 
  CommandLineOption.cc
  OptionParserImpl.cc
  OptionSettingHelpPrinter.cc
  OptionSettingImpl.cc)
set( HEADER_FILES 
  Argument.h
  ArgumentValue.h
  ArgumentValueFactory.h
  CommandLineOption.h
  OptionParser.h
  OptionParserImpl.h
  OptionSetting.h
  OptionSettingHelpPrinter.h
  OptionSettingImpl.h)
set( TEST_FILES
  OptionParserImpl_unittest.cc
  OptionSettingHelpPrinter_unittest.cc
  OptionSettingImpl_unittest.cc)

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} 
  "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
if (hetero_mark_test)
  add_executable(${FOLDER_NAME}_unittest 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_unittest gtest gtest_main)
  add_test(${FOLDER_NAME}_unittest ${FOLDER_NAME}_unittest)
endif (hetero_mark_test)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(CommandLineOption ${SRC_FILES} ${HEADER_FILES})
