#include <stdio.h>

#include <fstream>
#include <iostream>

#include <sys/stat.h>

using namespace std;

#include <json.hpp>
using json=nlohmann::json;

#include "../include/spdlog/spdlog.h"

// For NextWave Plugin
#include "nextwave_plugin.hpp"
//#pragma pack(push,1)
//#include "memory_layout.h"
//#pragma pack(pop) // restore previous setting

#include <chrono>
#include <thread>


#if _WIN64
#include <windows.h>
#include <strsafe.h>
#include "boost/interprocess/windows_shared_memory.hpp"
#else
#include <boost/interprocess/shared_memory_object.hpp>
// On unix, for dl* functions:
#include <dlfcn.h>
#endif

#include "nextwave_plugin.cpp"

#include "boost/interprocess/mapped_region.hpp"
using namespace boost::interprocess;

// Globals
#define BUF_SIZE 2048*2048

unsigned char buffer[BUF_SIZE];
unsigned int nCurrRing=0;
unsigned int width=0; // TODO: force square images for raw files
unsigned int height=0; // TODO: force square images for raw files

// UI Socket communication
#include <cstring>
#define CAMERA_SOCKET 50007
#include "socket.cpp"

// TODO: Replace with code in engine
inline uint64_t time_highres() {
#if _WIN64
  LARGE_INTEGER freq, now;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  return (float)now.QuadPart / freq.QuadPart * 1e6;
#else
  return 0; //TODO
#endif
}

std::string gfilename;

void read_file(std::string filename)
{
	FILE *fp;
  struct stat st;

  stat(filename.c_str(), &st);
  int file_size = st.st_size;

  // TODO: Temp wait until file all populated
  //while (file_size != 1000000) {
  while (file_size <992*992) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stat(filename.c_str(), &st);
    file_size = st.st_size;
  };

	fp = fopen(filename.c_str(),"rb");  // r for read, b for binary

  if (fp==NULL) {
    spdlog::error("Couldn't read file.");
  } else {

    fseek(fp, 0L, SEEK_END);
    width = sqrt(ftell(fp)); // TODO: only square allowed
    height=width;
    rewind(fp);

    fread(buffer,1,height*width,fp); // only bytes
    spdlog::info("Read {} {} {} {}",filename,height, width, file_size);
  }
  fclose(fp);

struct shmem_header* pShmem = (struct shmem_header*) shmem_region1.get_address();

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

 pShmem->lock = (uint8_t)0; // Everyone keep out until we are done!

 // Prime the connection
 //char *msg=socket_check(CAMERA_SOCKET);

  return; 
}

DECL init(char *params)
{
  plugin_access_shmem();

  json jdata = json::parse(params);
  gfilename=jdata["filename"];

  read_file(gfilename);

  spdlog::info("raw OK");

  return 0;
}

DECL process(char *params)
{
  char *msg=socket_check(CAMERA_SOCKET);

  //return 0; // Let the Python UI handle it.

  if (msg!=NULL) {
    spdlog::info("RAW: {}",msg);
    if (msg[0]=='E'){
        //if !strcmp(msg,"EXPO"){
        double dVal = atof(msg+2);
        spdlog::info("exposure: {}",dVal);
      }
  }

  read_file(gfilename);

	// DC NEW
	struct shmem_header* pShmem = (struct shmem_header*) shmem_region1.get_address();

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
		(unsigned char*)buffer,
		height*width);

	pShmem->lock = (uint8_t)0; // Keep out until we are done!

	nCurrRing += 1;
	if (nCurrRing >= NW_MAX_FRAMES)
    nCurrRing = 0;

	return 0;
};

DECL plugin_close(char *params)
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
