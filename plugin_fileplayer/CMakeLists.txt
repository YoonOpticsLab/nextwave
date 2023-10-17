cmake_minimum_required(VERSION 3.1...3.27)

project(plugin_fileplayer)

list(APPEND SOURCES
  plugin_fileplayer.cpp
  VERSION 1.0
  LANGUAGES CXX)


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(../include)

add_library(plugin_fileplayer SHARED plugin_fileplayer.cpp ../nextwave_plugin.hpp)
target_link_libraries( plugin_fileplayer ${OpenCV_LIBS} )
set_target_properties(plugin_fileplayer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ".."  )
