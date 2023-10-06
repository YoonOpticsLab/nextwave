cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_ximea)

list(APPEND SOURCES
    main.cpp
  VERSION 1.0
  LANGUAGES CXX)

set(XIMEA_DIR "D:/XIMEA")

include_directories(../)
include_directories(../common/include)
include_directories(../common/include/boost_1_83_0)

# include_directories(${XIMEA_DIR}/_libs/xiAPI)
include(add_xiapi.cmake)
#include(add_executable.cmake)


add_library(plugin_ximea SHARED main.cpp ../nextwave_plugin.hpp)
set_target_properties(plugin_ximea PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../debug"  )