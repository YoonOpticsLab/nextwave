cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_flir)

list(APPEND SOURCES
  Acquisition.cpp
  VERSION 1.0
  LANGUAGES CXX)

set(SPINNAKER_DIR "C:/Program Files/FLIR Systems/Spinnaker")

include_directories(${SPINNAKER_DIR}/include)

include_directories(../)
include_directories(../common/include)
include_directories(../common/include/boost_1_83_0)

# include_directories(${XIMEA_DIR}/_libs/xiAPI)
include(add_spinnaker.cmake)
#include(add_executable.cmake)


add_library(plugin_flir SHARED Acquisition.cpp ../nextwave_plugin.hpp)
#add_executable(plugin_flir Acquisition.cpp ../nextwave_plugin.hpp)

set_target_properties(plugin_flir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
# set_target_properties(plugin_flir PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../debug"  )