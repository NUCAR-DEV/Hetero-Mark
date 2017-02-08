set(FOLDER_NAME common)
set(SUBDIRECTORIES 
  benchmark
  command_line_option
  time_measurement
  cl_util
)

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
