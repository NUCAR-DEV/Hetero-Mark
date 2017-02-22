cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME src)
set(SUBDIRECTORIES
  common
  be
  aes
  fir
)

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
