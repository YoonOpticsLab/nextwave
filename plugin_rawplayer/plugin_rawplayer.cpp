#include <stdio.h>

#include <fstream>
#include <iostream>

using namespace std;

#include <json.hpp>
using json=nlohmann::json;

// For NextWave Plugin
#include "nextwave_plugin.hpp"
#pragma pack(push,1)
#include "memory_layout.h"
#pragma pack(pop) // restore previous setting

// Add this directory (right-click on project in solution explorer, etc.)
//#include "C:\Users\drcoates\Documents\code\nextwave\boost_1_83_0"
#include "boost/interprocess/windows_shared_memory.hpp"
#include "boost/interprocess/mapped_region.hpp"
using namespace boost::interprocess;

// Globals
#define BUF_SIZE 2048*2048
	
unsigned char buffer[BUF_SIZE];
unsigned int nCurrRing=0;
LARGE_INTEGER freq, now;

	
DECL init(char *params)
{
  json jdata = json::parse(params);
  std::string filename=jdata["filename"];

	FILE *fp;

	fp = fopen(filename.c_str(),"rb");  // r for read, b for binary

	fread(buffer,sizeof(buffer),1,fp);

QueryPerformanceFrequency(&freq);

  return 0;
}

   // https://stackoverflow.com/questions/8583308/timespec-equivalent-for-windows
inline uint64_t time_highres() {
	QueryPerformanceCounter(&now);
	return now.QuadPart / freq.QuadPart * 1e6;
}

		  
DECL process(char *params)
{
#if 1
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    mapped_region shmem_region{ shmem, read_write };

    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    mapped_region shmem_region2{ shmem2, read_write };
#endif //0
  const size_t width = 2048;
  const size_t height = 2048;
  
                    // DC NEW
                    struct shmem_header* pShmem = (struct shmem_header*) shmem_region.get_address();

                    pShmem->lock = (uint8_t)1; // Everyone keep out until we are done!

                    // Don't need to write these each time:
                    pShmem->header_version = (uint8_t)NW_HEADER_VERSION;
                    pShmem->dimensions[0] = (uint16_t)height;
                    pShmem->dimensions[1] = (uint16_t)width;
                    pShmem->dimensions[2] = (uint16_t)0;
                    pShmem->dimensions[3] = (uint16_t)0;
                    pShmem->datatype_code = (uint8_t)7;
                    pShmem->max_frames = (uint8_t)NW_MAX_FRAMES;

                    // For current frame:
                    pShmem->current_frame = (uint8_t)nCurrRing;
                    pShmem->timestamps[nCurrRing] = (uint8_t)NW_STATUS_READ;
                    pShmem->timestamps[nCurrRing] = time_highres();

                    memcpy( ((uint8_t *)(shmem_region2.get_address())+height*width*nCurrRing),
                        (void*)buffer,
                        height*width);

                    pShmem->lock = (uint8_t)0; // Keep out until we are done!
					
                    nCurrRing += 1;
                    if (nCurrRing >= 32) nCurrRing = 0;
  
	return 0;
};

DECL close(char *params)
{
  std::cout <<"P1 close\n" ;
  return 0;
};

DECL set_params(char* settings_as_json)
{
  std::cout <<"P1 set_params " << settings_as_json << "\n";
  return 0;
};
 
DECL get_info(char* which_info)
{
  std::cout <<"P1 get_info";
  return 0;
};



