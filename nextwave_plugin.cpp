#include <json.hpp>
using json=nlohmann::json;
#include "../include/spdlog/spdlog.h"

#if _WIN64
#include <windows.h>
#include <strsafe.h>
#include "boost/interprocess/windows_shared_memory.hpp"
#else
#include <boost/interprocess/shared_memory_object.hpp>
// On unix, for dl* functions:
#include <dlfcn.h>
#endif

#include <boost/interprocess/mapped_region.hpp>

using namespace boost::interprocess;

// Put this in a file in order to declare local vars to access shmem
#if _WIN64
windows_shared_memory shmem1;
windows_shared_memory shmem2;
windows_shared_memory shmem3;
mapped_region shmem_region1;
mapped_region shmem_region2;
mapped_region shmem_region3;
#else
shared_memory_object shmem1;
shared_memory_object shmem2;
shared_memory_object shmem3;
mapped_region shmem_region1;
mapped_region shmem_region2;
mapped_region shmem_region3;
#endif

struct shmem_header* gpShmemHeader;
struct shmem_boxes_header* gpShmemBoxes;
 

// For NextWave Plugin
#pragma pack(push,1)
#include "memory_layout.h"
#include "layout_boxes.h"
#pragma pack(pop) // restore previous setting

#if _WIN64
// Windows: statically link module into .EXE
#include <windows.h>
#define PLUGIN_API(prefix,which,params) extern "C"  __declspec(dllexport) int prefix##_##which(params)
//#define PLUGIN_API(prefix,which,params) extern "C"  __declspec(dllexport) int which(params)
#else
// Linux: Put in .DLL/.SO
#define PLUGIN_API(prefix,which,params) extern "C" int which(params)
#endif

static void plugin_access_shmem( void ) {
#if _WIN64
  shmem1=windows_shared_memory(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
  shmem2=windows_shared_memory(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
  shmem3=windows_shared_memory(open_or_create, SHMEM_BUFFER_NAME_BOXES, read_write, (size_t)sizeof(shmem_boxes_header));
#else
  shmem1=shared_memory_object(open_or_create, SHMEM_HEADER_NAME, read_write);
  shmem2=shared_memory_object(open_or_create, SHMEM_BUFFER_NAME, read_write);
  shmem3=shared_memory_object(open_or_create, SHMEM_BUFFER_NAME_BOXES, read_write);
#endif

  shmem_region1=mapped_region(shmem1, read_write);
  shmem_region2=mapped_region(shmem2, read_write);
  shmem_region3=mapped_region(shmem3, read_write);

#if _WIN64
#else
  shmem1.truncate((size_t)SHMEM_HEADER_SIZE);
  shmem2.truncate((size_t)SHMEM_BUFFER_SIZE);
  shmem3.truncate((size_t)sizeof(shmem_boxes_header));
#endif

  gpShmemHeader= (struct shmem_header*) shmem_region1.get_address();
  gpShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
}
