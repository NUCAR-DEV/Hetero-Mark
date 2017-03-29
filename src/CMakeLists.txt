cmake_minimum_required(VERSION 3.2)

set(FOLDER_NAME src)
set(SUBDIRECTORIES
  common
  aes
  be
  bs
  ep
  fir
  ga
  hist
)

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
endforeach(subdir)
