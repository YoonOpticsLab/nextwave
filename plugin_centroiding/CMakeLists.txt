cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_centroiding)

find_package(ArrayFire REQUIRED)

list(APPEND SOURCES
  plugin_centroiding.cpp
  VERSION 1.0
  LANGUAGES CXX)

include_directories(../include)
include_directories(..)
include_directories(include/boost_1_83_0)
include_directories(include)
include_directories(${ArrayFire_INCLUDE_DIRS})

add_library(plugin_centroiding SHARED plugin_centroiding.cpp ../nextwave_plugin.hpp)
set_target_properties(plugin_centroiding PROPERTIES LIBRARY_OUTPUT_DIRECTORY "c:/nextwave/build"  )
set_target_properties(plugin_centroiding PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
target_link_libraries(plugin_centroiding ${ArrayFire_LIBRARIES})