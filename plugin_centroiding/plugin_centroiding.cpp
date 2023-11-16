#include <stdio.h>

#include <fstream>
#include <iostream>

#include <arrayfire.h>
 
 
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

uint64_t xcoord[BUF_SIZE];
uint64_t ycoord[BUF_SIZE];

af::array im; // = af::constant(0, 2048, 2048);

// https://stackoverflow.com/questions/8583308/timespec-equivalent-for-windows
inline uint64_t time_highres() {
	QueryPerformanceCounter(&now);
	return now.QuadPart / freq.QuadPart * 1e6;
}
	
DECL init(char *params)
{
	QueryPerformanceFrequency(&freq);
	long long t1=time_highres();
	
    // Generate 10,000 random values
// https://stackoverflow.com/questions/49692531/transfer-data-from-arrayfire-arrays-to-armadillo-structures
// https://github.com/arrayfire/arrayfire/issues/1405;


    // Sum the values and copy the result to the CPU:
    //double sum = af::sum<float>(a);
	long long t2=time_highres();
 
    //printf("sum: %g TIME=%ld\n", sum, (long) (t2-t1) );

  return 0;
}



		  
DECL process(char *params)
{
#if 1
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    mapped_region shmem_region{ shmem, read_write };

    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    mapped_region shmem_region2{ shmem2, read_write };

    windows_shared_memory shmem3(open_or_create, SHMEM_BUFFER_NAME2, read_write, (size_t)SHMEM_BUFFER_SIZE2);
    mapped_region shmem_region3{ shmem3, read_write };
#endif //0

	struct shmem_header* pShmem = (struct shmem_header*) shmem_region.get_address();
	uint16_t nCurrRing = pShmem->current_frame;
	uint16_t height = pShmem->dimensions[0];
	uint16_t width = pShmem->dimensions[1];
	memcpy((void*)buffer,
	((uint8_t *)(shmem_region2.get_address())+height*width*nCurrRing),    
		height*width);
		
	af::array im=af::array(width, height, buffer, afDevice);
  
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



