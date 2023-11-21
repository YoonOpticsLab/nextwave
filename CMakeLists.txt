# CMake simple example

## [main]

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1...3.27)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  minimal_plugin
  VERSION 1.0
  LANGUAGES CXX)

# https://stackoverflow.com/questions/29657195/how-to-test-if-cmake-found-a-library-with-find-library
if (UNIX) # For Dan
  SET(GCC_OPT "")
  SET(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} ${GCC_OPT} )
  SET(CMAKE_EXE_LINKER_FLAGS  ${CMAKE_EXE_LINKER_FLAG} ${GCC_OPT} )
endif()

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)


#include_directories(common/include)
#include_directories(d:/code/boost_1_83_0)
include_directories(include/boost_1_83_0)
include_directories(include)

add_library(plugin_test1 SHARED plugin_test1.cpp nextwave_plugin.hpp)
add_library(plugin_test2 SHARED plugin_test2.cpp nextwave_plugin.hpp)

## [main]

add_subdirectory(plugin_ximea)
# add_subdirectory(plugin_fileplayer)
add_subdirectory(plugin_rawplayer)

#add_subdirectory(plugin_flir)
add_subdirectory(plugin_centroiding)

add_executable(engine engine.cpp)

if (WIN32)
  target_link_libraries(engine plugin_centroiding.lib) # TODO: need .lib?
else()
  target_link_libraries(engine plugin_rawplayer)
  target_link_libraries(engine plugin_centroiding)
endif()
