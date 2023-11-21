cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_rawplayer)

list(APPEND SOURCES
  plugin_rawplayer.cpp
  VERSION 1.0
  LANGUAGES CXX)

include_directories(../include)
include_directories(..)
include_directories(include/boost_1_83_0)
include_directories(include)

add_library(plugin_rawplayer SHARED plugin_rawplayer.cpp ../nextwave_plugin.hpp)
if (WIN32)
  set_target_properties(plugin_rawplayer PROPERTIES LIBRARY_OUTPUT_DIRECTORY "c:/nextwave/build"  )
  set_target_properties(plugin_rawplayer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
else()
  set_target_properties(plugin_rawplayer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "../../build"
    RUNTIME_OUTPUT_DIRECTORY "../../build"  )
endif()
