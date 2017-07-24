file(GLOB_RECURSE ALL_SOURCE_FILES 
	RELATIVE ${CMAKE_SOURCE_DIR}
	src/*.cc src/*.cpp src/*.cu src/*.h)
file(GLOB_RECURSE ALL_KERNEL_FILES
	RELATIVE ${CMAKE_SOURCE_DIR}
	src/*.cl)

find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
	message("Python not found")
endif()

find_file(CLANG_FORMAT 
	NAMES clang-format 
	DOC "Clang format tool")
if(NOT CLANG_FORMAT)
	message ("clang-format not found")
endif()

add_custom_target(check
	COMMAND "${CMAKE_COMMAND}" -E chdir
	"${CMAKE_SOURCE_DIR}"
	clang-format -style=file -i
	${ALL_SOURCE_FILES}
	DEPENDS ${AlL_SOURCE_FILES}
	COMMENT "Linting source code"
	VERBATIM)
