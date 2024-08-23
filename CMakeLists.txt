# CMake simple example

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.1...3.27)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  nextwave_engine
  VERSION 1.0
  LANGUAGES CXX)

# https://stackoverflow.com/questions/29657195/how-to-test-if-cmake-found-a-library-with-find-library
if (UNIX) # For Dan
  SET(GCC_OPT "")
  SET(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} ${GCC_OPT} )
  SET(CMAKE_EXE_LINKER_FLAGS  ${CMAKE_EXE_LINKER_FLAG} ${GCC_OPT} )
else() # For Windows, need AF here
	find_package(ArrayFire REQUIRED)

	include_directories(${ArrayFire_INCLUDE_DIRS})

	# - WIN32_LEAN_AND_MEAN & VC_EXTRALEAN reduces the number of
	#   windows headers being included.
	# - NOMINMAX is required for ArrayFire code that uses
	#   functions af::min & af::max. Having a namespace doesn't help also.
	add_definitions(-DWIN32_LEAN_AND_MEAN -DVC_EXTRALEAN -DNOMINMAX -DAF_TRACE -DAF_PRINT_ERRORS)
	#  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)

endif()

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)

# Download this to include directory:
# https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp

# include_directories(include/boost_1_83_0)
include_directories(include)

add_library(plugin_test1 SHARED plugin_test1.cpp nextwave_plugin.hpp)
add_library(plugin_test2 SHARED plugin_test2.cpp nextwave_plugin.hpp)

if (UNIX) 
else()
    add_subdirectory(plugin_alpao)
    add_subdirectory(plugin_flir)
endif()

# add_subdirectory(plugin_ximea)
# add_subdirectory(plugin_fileplayer)
add_subdirectory(plugin_rawplayer)
add_subdirectory(plugin_centroiding)

if (WIN32)
    include_directories(.)
    include_directories(..)
	add_executable(engine engine.cpp plugin_centroiding/plugin_centroiding.cpp)
	#add_executable(engine engine.cpp plugin_centroiding/plugin_centroiding.cpp)
	#target_link_libraries(engine ${ArrayFire_Unified_LIBRARIES} )
	target_link_libraries(engine ArrayFire::af)
	#target_link_libraries(engine plugin_centroiding.lib) # TODO: need .lib?
else()
	add_executable(engine engine.cpp)
	target_link_libraries(engine plugin_rawplayer)
	target_link_libraries(engine plugin_centroiding)
endif()
