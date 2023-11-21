#include <arrayfire.h>

#include <stdio.h>

#include <fstream>
#include <iostream>
  
#include <crtdbg.h> 
  
using namespace std;

#include <json.hpp>
using json=nlohmann::json;

#include "../spdlog/spdlog.h"

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
#define BUF_SIZE (2048*2048)
	
unsigned char buffer[BUF_SIZE];
unsigned int nCurrRing=0;
LARGE_INTEGER freq, now;
LARGE_INTEGER t2, t1;

uint64_t xcoord[BUF_SIZE];
uint64_t ycoord[BUF_SIZE];

//af::array im= af::constant(0, 2048, 2048);

// https://stackoverflow.com/questions/8583308/timespec-equivalent-for-windows
inline double time_highres() {
	QueryPerformanceCounter(&now);
	return now.QuadPart / freq.QuadPart * 1e6;
}
	
DECL init(char *params)
{
	//af::setBackend(AF_BACKEND_CUDA);
	
	QueryPerformanceFrequency(&freq);
	
    // Generate 10,000 random values
// https://stackoverflow.com/questions/49692531/transfer-data-from-arrayfire-arrays-to-armadillo-structures
// https://github.com/arrayfire/arrayfire/issues/1405;


    // Sum the values and copy the result to the CPU:
    //double sum = af::sum<float>(a);
	//
 
    //printf("sum: %g TIME=%ld\n", sum, (long) (t2-t1) );

  return 0;
}


//#define WIDTH 2048
//#define HEIGHT 2048
#define BOX_SIZE 80
#define NBOXES 25*25
float fbuffer[BUF_SIZE];
int nbuffer[BUF_SIZE];

  
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
	
#if 1
	memcpy((void*)buffer,
	((const void *)(shmem_region2.get_address())), //+height*width*nCurrRing),    
		height*width);
#endif //0		
	//af::array im= af::constant(0, 2048, 2048);
	//auto t1=time_highres();

	spdlog::info("Dims: {} {}", width, height) ;



	af::array im;
	int WIDTH=width;
	int HEIGHT = height;
	

	af::array im_xcoor;
	af::array im_ycoor;

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

	af::array box_x;
	af::array box_y;
	af::array box_idx1;

	af::array cen_x;
	af::array cen_y;

	af::array weighted_x;
	af::array weighted_y;

	af::array sums_x;
	af::array sums_y;

	
	//int nbuffer[]={0,1,2,3,4,5,6,7,8,9};
	//#if 0

	for (int ibox=0; ibox<NBOXES; ibox++) {
		//for (int x=0; x<10; x++) 
		for (int y=0; y<BOX_SIZE; y++) {
			nbuffer[ibox*BOX_SIZE+y]=ibox*BOX_SIZE+y; //0+940-80*5;
		}
	}
	//#endif //0
	box_x = af::array(BOX_SIZE,NBOXES,nbuffer);

	for (int ibox=0; ibox<NBOXES; ibox++) {
		//for (int x=0; x<10; x++) 
		for (int y=0; y<BOX_SIZE; y++) {
			nbuffer[ibox*BOX_SIZE+y]=ibox*BOX_SIZE+y; //+940-80*4;
		}
	}
	//#endif //0
	box_y = af::array(BOX_SIZE,NBOXES,nbuffer);
	
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
	
	// "Real" processing loop
	QueryPerformanceCounter(&t1);

	box_idx1 = af::array(BOX_SIZE*BOX_SIZE, NBOXES, nbuffer );
	
	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			fbuffer[y*width+x]=(float)buffer[y*width+x];
		}
	}
	im=af::array(width, height, fbuffer);		
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
	//xbetter = diag(xbetter, 0 );
	
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

	QueryPerformanceCounter(&t2);

	spdlog::info("Maxx: {}", (float)af::max<float>(sums_x));
	spdlog::info("Maxy: {}", (float)af::max<float>(sums_y));
	//auto t2=time_highres();
	
	//spdlog::info("AF: {}", (float)af::max<float>(im) ) ;
	spdlog::info("Time: {}", float(t2.QuadPart-t1.QuadPart)/freq.QuadPart) ;
   
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



