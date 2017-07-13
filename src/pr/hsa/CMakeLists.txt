cmake_minimum_required(VERSION 3.2.0)

set(FOLDER_NAME pr_hsa )
set(SUBDIRECTORIES )
set(SRC_FILES
	main.cc
	pr_hsa_benchmark.h
	pr_hsa_benchmark.cc
	)
set(HEADER_FILES )
set(EXTRA_FILES
	kernels.cl
	)

if(have_snack AND HSA_RUNTIME)
	set_source_files_properties(
		${FOLDER_NAME}_kernels.o
		PROPERTIES GENERATED TRUE
		)

	add_custom_target(
		${FOLDER_NAME}_kernels.o
		COMMAND ${CMAKE_COMMAND} -E copy 
			${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl
			${CMAKE_CURRENT_BINARY_DIR}/kernels.cl
		COMMAND ${SNACK} -c kernels.cl
		COMMAND mv kernels.o ${FOLDER_NAME}_kernels.o
		DEPENDS kernels.cl
		COMMENT "Using snack to generate kernel library"
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		VERBATIM
		)


	add_executable(${FOLDER_NAME} ${FOLDER_NAME}_kernels.o ${HEADER_FILES} ${SRC_FILES})
	target_link_libraries(${FOLDER_NAME}
		pr time_measurement command_line_option benchmark ${HSA_RUNTIME})
	add_dependencies(${FOLDER_NAME} ${FOLDER_NAME}_kernels.o)
endif(have_snack AND HSA_RUNTIME)
