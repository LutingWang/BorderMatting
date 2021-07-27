cmake_minimum_required (VERSION 3.0)
project (RecognitionTools)
set (CMAKE_CXX_STANDARD 11)

find_package (OpenCV REQUIRED)
message (STATUS "This is opencv include dirs " ${OpenCV_INCLUDE_DIRS})
message (STATUS "This is opencv libs " ${OpenCV_LIBS})

include_directories (include)

aux_source_directory (src SRCS)
add_executable (grabcut ${SRCS})
target_link_libraries (grabcut ${OpenCV_LIBS})
