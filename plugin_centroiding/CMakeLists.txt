cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_centroiding)

find_package(ArrayFire REQUIRED)

set(CMAKE_CXX_STANDARD 14)

list(APPEND SOURCES
  plugin_centroiding.cpp
  VERSION 1.0
  LANGUAGES CXX)

include_directories(../include)
include_directories(..)
include_directories(include/boost_1_83_0)
include_directories(include)
include_directories(${ArrayFire_INCLUDE_DIRS})

if(WIN32)
  # - WIN32_LEAN_AND_MEAN & VC_EXTRALEAN reduces the number of
  #   windows headers being included.
  # - NOMINMAX is required for ArrayFire code that uses
  #   functions af::min & af::max. Having a namespace doesn't help also.
  add_definitions(-DWIN32_LEAN_AND_MEAN -DVC_EXTRALEAN -DNOMINMAX -DAF_TRACE -DAF_PRINT_ERRORS)
#  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
endif()

#https://discourse.cmake.org/t/windows-libraries-find-modules-and-target-runtime-dlls-re-re-revisited/4286
add_library(plugin_centroiding SHARED plugin_centroiding.cpp ../nextwave_plugin.hpp)
target_link_libraries(plugin_centroiding ${ArrayFire_Unified_LIBRARIES} )
if (WIN32)
  set_target_properties(plugin_rawplayer PROPERTIES LIBRARY_OUTPUT_DIRECTORY "c:/nextwave/build"  )
  set_target_properties(plugin_rawplayer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
else()
  set_target_properties(plugin_centroiding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "../../build"
    RUNTIME_OUTPUT_DIRECTORY "../../build"  )
endif()
#target_link_libraries(vectorize_cuda ArrayFire::afcuda)
