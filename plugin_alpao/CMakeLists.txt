cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_alpao)

list(APPEND SOURCES
  plugin_alpao.cpp
  VERSION 1.0
  LANGUAGES CXX)

include_directories(../include)
include_directories(..)
include_directories(include)
include_directories(include/boost_1_83_0)
include_directories(../sdk/include)
include_directories(sdk/include)

#link_libraries(plugin_alpao ASDK.dll )
#link_directories(c:/nextwave/sdk/lib/x64)
#target_link_libraries(target_name C:/Users/username/libabcd.
#https://discourse.cmake.org/t/windows-libraries-find-modules-and-target-runtime-dlls-re-re-revisited/4286

add_library(plugin_alpao SHARED plugin_alpao.cpp)
link_libraries(ASDK)
target_link_libraries(plugin_alpao C:/nextwave/sdk/lib/x64/asdk.lib)
	
set_target_properties(plugin_alpao PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
	
#if (WIN32)
#  set_target_properties(plugin_alpao PROPERTIES LIBRARY_OUTPUT_DIRECTORY "c:/nextwave/build"  )
#  set_target_properties(plugin_alpao PROPERTIES RUNTIME_OUTPUT_DIRECTORY "c:/nextwave/build"  )
#else()
#  set_target_properties(plugin_alpao PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY "../../build"
#    RUNTIME_OUTPUT_DIRECTORY "../../build"  )
#endif()
