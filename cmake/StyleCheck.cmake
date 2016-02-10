file(GLOB_RECURSE ALL_SOURCE_FILES 
    RELATIVE ${CMAKE_SOURCE_DIR}
    src/*.cc src/*.cpp src/*.h)

find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
  message("Python not found")
endif()

find_file(CPP_LINT_PY 
  NAMES cpplint.py 
  HINTS ${CMAKE_SOURCE_DIR}
  DOC "Google cpp style scan program.")
if(NOT CPP_LINT_PY)
  message ("cpplint.py not found")
endif()

find_file(CLANG_FORMAT 
  NAMES clang-format 
  HINTS ${CMAKE_SOURCE_DIR}
  DOC "Clang format tool")
if(NOT CLANG_FORMAT)
  message ("clang-format not found")
endif()

add_custom_target(check
  COMMAND "${CMAKE_COMMAND}" -E chdir
    "${CMAKE_SOURCE_DIR}"
    ${CMAKE_SOURCE_DIR}/clang-format -style=Google -i
    ${ALL_SOURCE_FILES}
  COMMAND "${CMAKE_COMMAND}" 
    -E chdir "${CMAKE_SOURCE_DIR}"
    ${CMAKE_SOURCE_DIR}/cpplint.py
    ${ALL_SOURCE_FILES}
  DEPENDS ${AlL_SOURCE_FILES}
  COMMENT "Linting source code"
  VERBATIM)
