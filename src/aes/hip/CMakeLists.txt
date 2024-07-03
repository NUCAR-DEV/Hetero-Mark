set_source_files_properties(aes_hip_benchmark.cpp PROPERTIES LANGUAGE HIP)
add_executable(aes_hip main.cc aes_hip_benchmark.cpp aes_hip_benchmark.h)

target_link_libraries(aes_hip PUBLIC aes time_measurement command_line_option
        benchmark hip::host hip::device)

