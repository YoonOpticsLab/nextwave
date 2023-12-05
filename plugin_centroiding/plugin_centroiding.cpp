#include <arrayfire.h>

#include <stdio.h>

#include <fstream>
#include <iostream>

//#include <crtdbg.h>

using namespace std;

#include <json.hpp>
using json=nlohmann::json;

#include "../include/spdlog/spdlog.h"

// For NextWave Plugin
#include "nextwave_plugin.hpp"
#pragma pack(push,1)
#include "memory_layout.h"
#pragma pack(pop) // restore previous setting

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

#define VERBOSE 0

// Globals
#define BUF_SIZE (2048*2048)

unsigned char buffer[BUF_SIZE];
unsigned int nCurrRing=0;
//LARGE_INTEGER t2, t1;

uint64_t xcoord[BUF_SIZE];
uint64_t ycoord[BUF_SIZE];

// https://stackoverflow.com/questions/8583308/timespec-equivalent-for-windows
inline double time_highres() {
#if _WIN64
  LARGE_INTEGER freq, now;
	QueryPerformanceCounter(&now);
	return now.QuadPart / freq.QuadPart * 1e6;
#else
  return 0;
#endif
}

af::array im_xcoor;
af::array im_ycoor;

af::array box_x;
af::array box_y;
af::array box_idx1;

af::array im;
af::array cen_x;
af::array cen_y;

af::array weighted_x;
af::array weighted_y;

af::array sums_x;
af::array sums_y;

//#define WIDTH 2048
//#define HEIGHT 2048
#define BOX_SIZE 80
#define NBOXES 25*25
float fbuffer[BUF_SIZE];
int nbuffer[BUF_SIZE];

DECL init(char *params)
{
  try {
    af::setBackend(AF_BACKEND_CPU);
    spdlog::info("Set AF_BACKEND_CPU");
  } catch (...){
    spdlog::error("Couldn't load AF BACKEND");
  }

  int width=1000; // TODO: fixme
  int height=1000;

	int WIDTH=width;
	int HEIGHT = height;
	for (int x=0; x<WIDTH; x++) {
		for (int y=0; y<HEIGHT; y++) {
			fbuffer[y*WIDTH+x]=(float)x;
		}
	}
	im_xcoor=af::array(WIDTH, HEIGHT, fbuffer);
	for (int x=0; x<WIDTH; x++) {
		for (int y=0; y<HEIGHT; y++) {
			fbuffer[y*WIDTH+x]=(float)y;
		}
	}

	im_ycoor=af::array(WIDTH, HEIGHT, fbuffer);

	for (int ibox=0; ibox<NBOXES; ibox++) {
		for (int y=0; y<BOX_SIZE; y++) {
			nbuffer[ibox*BOX_SIZE+y]=ibox*BOX_SIZE+y; //0+940-80*5;
		}
	}
	box_x = af::array(BOX_SIZE,NBOXES,nbuffer);

	for (int ibox=0; ibox<NBOXES; ibox++) {
		for (int y=0; y<BOX_SIZE; y++) {
			nbuffer[ibox*BOX_SIZE+y]=ibox*BOX_SIZE+y; //+940-80*4;
		}
	}
	box_y = af::array(BOX_SIZE,NBOXES,nbuffer);

  // Build bogus boxes:
	for (int ibox=0; ibox<NBOXES; ibox++) {
		int ibox_x = ibox % 20;
		int ibox_y = int(float(ibox) / 20);
		for (int y=0; y<BOX_SIZE; y++)  {
			for (int x=0; x<BOX_SIZE; x++) {
				int posx=ibox_x*BOX_SIZE+x+25 + rand() % 20;        ; // +25 just to approx. center spots in test img. TODO!!
				int posy=ibox_y*BOX_SIZE+y+25 + rand() % 20;
				nbuffer[ibox*BOX_SIZE*BOX_SIZE+y*BOX_SIZE+x]=posy*WIDTH+posx; //+940-80*4;
			}
		}
	}
	box_idx1 = af::array(BOX_SIZE*BOX_SIZE, NBOXES, nbuffer );

  return 0;
}
// https://stackoverflow.com/questions/49692531/transfer-data-from-arrayfire-arrays-to-armadillo-structures
// https://github.com/arrayfire/arrayfire/issues/1405;
int find_cendroids_af(unsigned char *buffer, int width, int height) {

	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			fbuffer[y*width+x]=(float)buffer[y*width+x];
		}
	}
	im=af::array(width, height, fbuffer);

#if VERBOSE
  printf("%f\n",(float)af::max<float>(im) );
#endif
	im /= 255.0;

	af::dim4 new_dims(BOX_SIZE*BOX_SIZE,NBOXES); // AF stacks the boxes, so do this to reshape for easier sum reduction

	weighted_x = (im) * im_xcoor;
	weighted_y = (im) * im_ycoor;

	af::array seqX=(box_x);
	af::array seqY=(box_y);
	af::array seq1=box_idx1;

	int doprint=0;
	if (doprint*0)
		af_print(seq1);

	af::array cool = weighted_x(seq1); //weighted_x or im_xcoor for debug
	if (doprint*0)
		af_print(cool );
	af::array xbetter = moddims( cool, new_dims );

	if (doprint*0)
		af_print(xbetter );

	cool = weighted_y(seq1);
	//af_print(cool );
	af::array ybetter = af::moddims( cool, new_dims );
	//af_print( ybetter );
	if (doprint*0)
		af_print(ybetter );

	af::array im2 = im(seq1);
	af::array im_better = af::moddims( im2, new_dims );

	af::array sums = af::sum( im_better, 0);
	//sums = af::sum( sums, 1);
	//sums /= sums;
	sums_x = af::sum( xbetter, 0) / sums;
	sums_y = af::sum( ybetter, 0) / sums;
	//sums1 = af::mean( sums1, 1);

	if (0) {
		af_print( sums_x );
		af_print( sums_y );
	}
#if 0
#endif //0

	//QueryPerformanceCounter(&t2); // TODO

#if VERBOSE
	spdlog::info("Maxx: {}", (float)af::max<float>(sums_x));
	spdlog::info("Maxy: {}", (float)af::max<float>(sums_y));
#endif

	//spdlog::info("Time: {}", float(t2.QuadPart-t1.QuadPart)/freq.QuadPart) ; // TODO
  return 0;
}

DECL process(char *params)
{

  using namespace boost::interprocess;
#if _WIN64
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    windows_shared_memory shmem3(open_or_create, SHMEM_BUFFER_NAME2, read_write, (size_t)SHMEM_BUFFER_SIZE2);
#else
    shared_memory_object shmem(open_or_create, SHMEM_HEADER_NAME, read_write);
    shmem.truncate((size_t)SHMEM_HEADER_SIZE);
    shared_memory_object shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write);
    shmem2.truncate((size_t)SHMEM_BUFFER_SIZE);
    shared_memory_object shmem3(open_or_create, SHMEM_BUFFER_NAME2, read_write);
    shmem3.truncate((size_t)SHMEM_BUFFER_SIZE2);
#endif

  // Common to both OSes:
  mapped_region shmem_region{ shmem, read_write };
  mapped_region shmem_region2{ shmem2, read_write };
  mapped_region shmem_region3{ shmem3, read_write };

	struct shmem_header* pShmem = (struct shmem_header*) shmem_region.get_address();
	uint16_t nCurrRing = pShmem->current_frame;
	uint16_t height = pShmem->dimensions[0];
	uint16_t width = pShmem->dimensions[1];

	memcpy((void*)buffer,
         ((const void *)(shmem_region2.get_address()))+height*width*nCurrRing, height*width);

#if VERBOSE
	spdlog::info("Dims: {} {} {}", width, height, int(buffer[0])) ;
#endif

  find_cendroids_af(buffer, width, height);
	return 0;
};

DECL plugin_close(char *params)
{
  std::cout <<"CEN close\n" ;
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



