
#include <arrayfire.h>

#include <stdio.h>

#include <fstream>
#include <iostream>

//#include <crtdbg.h>

using namespace std;

#include <json.hpp>
using json=nlohmann::json;

#include "../include/spdlog/spdlog.h"

#include "nextwave_plugin.cpp"

// UI Socket communication
#include <cstring>
#define CENTROIDING_SOCKET 50008
#include "socket.cpp"

#define VERBOSE 0

// Globals
#define BUF_SIZE (2048*2048)

unsigned char buffer[BUF_SIZE];

uint8_t bDoSubtractBackground=0;
uint8_t bDoSetBackground=0;
uint8_t bDoReplaceSubtracted=0;

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

// Make a class to store AF arrays across process calls.
// Thus memory isn't destroyed/erased constantly (causing Garbage Collection issues).
// There are members having to do with the boxes (which are updated sporadically)
// and members which are created with each frame. (TODO: document the lifespan and purpose of each array)
class af_instance {
public: 

  af_instance() {
  };

  ~af_instance() {
  };

  af::array im;
  af::array im_background;
  af::array im_temp;

  af::array im_xcoor;
  af::array im_ycoor;

  af::array box_x;
  af::array box_y;

  af::array cen_x;
  af::array cen_y;

  af::array weighted_x;
  af::array weighted_y;

  af::array sums_x;
  af::array sums_y;

  af::array seq1;

  af::dim4 new_dims;//(BOX_SIZE*BOX_SIZE,NBOXES); // AF stacks the boxes, so do this to reshape for easier sum reduction

  // These are "temp" & overwritten each time in the loop, but put here to avoid GC, etc. :
  af::array im2;
  af::array sums;

  af::array atemp;
  af::array x_reshape;
  af::array y_reshape;
};

class af_instance *gaf;

float fbuffer[BUF_SIZE];
int nbuffer[BUF_SIZE];

uint16_t num_boxes;
double pixel_um;
double box_size_float;
int box_size; 
int box_size_int; // TODO remove me

double pupil_radius_pixels;

int width;
int height;

float local_refs[MAX_BOXES];

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

void init_buffers(int width, int height, int box_size, int nboxes) {
  // Precompute indexing arrays for weighted average
	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			fbuffer[x*width+y]=(float)x;
		}
	}
	gaf->im_xcoor=af::array(width, height, fbuffer); // wedth and hight sietched? TODO

	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			fbuffer[x*width+y]=(float)y;
		}
	}
	gaf->im_ycoor=af::array(width, height, fbuffer);
  gaf->new_dims=af::dim4(box_size*box_size,nboxes); // AF stacks the boxes, so do this to reshape for easier sum reduction
}

void read_boxes(int width) {

  // TODO: is dynamically changing the size allowed?
  struct shmem_header* pShmem = (struct shmem_header*) shmem_region1.get_address();
  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();

  num_boxes = pShmemBoxes->num_boxes;
  pixel_um = pShmemBoxes->pixel_um;
  box_size_float = pShmemBoxes->box_um/pixel_um;
  box_size = (int)(box_size_float+0.5); // round up to nearest integer
  box_size_int = box_size; // TODO:remove me

  pupil_radius_pixels = pShmemBoxes->pupil_radius_um/pixel_um;

  height = pShmem->dimensions[0];
  width = pShmem->dimensions[1];

  // TODO: this will be slow to do every time.
  init_buffers(width, height, box_size_int, num_boxes);

#if 0
  // DEbugging
  //float x_ref0 = pShmemBoxes->reference_x[0];
  //float y_ref0 = pShmemBoxes->reference_y[0];
 
	//spdlog::info("RB0 {} {}", x_ref0, y_ref0);

	//af_print_mem_info("message", -1);

	//spdlog::info("RBf {} {} {}", box_size,num_boxes,pShmemBoxes->reference_x[0]);
	try {
		memcpy(local_refs, pShmemBoxes->reference_x, sizeof(float) * MAX_BOXES);
		//af::array box_x = af::array(box_size, num_boxes, local_refs); // pShmemBoxes->reference_x);
    } catch (af::exception &e) { fprintf(stderr, "%s\n", e.what()); }

	spdlog::info("RBf {} {} {}", box_size,num_boxes,pShmemBoxes->reference_x[0]);
	//gaf->box_y = af::array(box_size,num_boxes,pShmemBoxes->reference_y);
	spdlog::info("RB1");
#endif //0

  // Each box will have a set of 1D indices into its members
	for (int ibox=0; ibox<num_boxes; ibox++) {
		int ibox_x = pShmemBoxes->box_x[ibox]-box_size/2;
		int ibox_y = pShmemBoxes->box_y[ibox]-box_size/2;
		for (int y=0; y<box_size; y++)  {
			for (int x=0; x<box_size; x++) {
        int posx=ibox_x+x;
        int posy=ibox_y+y;
        int idx_1d=posx*width+posy; // Column major // TODO: height
        //if (ibox==0) {
        //printf("%d ",idx_1d);
        //}
				//nbuffer[ibox*box_size_int*box_size_int+x*box_size_int+y]=idx_1d;
				nbuffer[ibox*box_size_int*box_size_int+x*box_size_int+y]=idx_1d;
			}
		}
	}
	gaf->seq1 = af::array(box_size*box_size, num_boxes, nbuffer );

  //spdlog::info("boxes: #={} size={} pupil={}", num_boxes, box_size, pupil_radius_pixels );
  //spdlog::info("x0={} y0={}", x_ref0, y_ref0 );

#if VERBOSE
  printf("%f\n",(float)af::max<float>(gaf->seq1) );
  printf("mn1: %f\n",(float)af::min<float>(gaf->seq1(af::span,1)) );
  printf("mn2: %f\n",(float)af::min<float>(gaf->seq1(af::span,2)) );
  printf("mx1: %f\n",(float)af::max<float>(gaf->seq1(af::span,1)) );
  printf("mx2: %f\n",(float)af::max<float>(gaf->seq1(af::span,2)) );
  printf("count1: %f\n",(float)af::count<float>(gaf->seq1(af::span,1)) );
#endif

	spdlog::info("CEN Init: {} {} {} {} {}", num_boxes, pixel_um, box_size, box_size_int, pupil_radius_pixels);
}

PLUGIN_API(centroiding,init,char *params)
{
  try {
    af::setBackend(AF_BACKEND_CUDA);
    spdlog::info("Set AF_BACKEND_CUDA");
  } catch (...){
      try {
      af::setBackend(AF_BACKEND_CPU);
      spdlog::info("Set AF_BACKEND_CPU");
    } catch (...){
      spdlog::error("Couldn't load any AF_BACKEND!!");
    }
  }

  plugin_access_shmem();
  gaf=new af_instance();
  return 0;
}

// https://stackoverflow.com/questions/49692531/transfer-data-from-arrayfire-arrays-to-armadillo-structures
// https://github.com/arrayfire/arrayfire/issues/1405;
int find_cendroids_af(unsigned char *buffer, int width, int height) {

	gaf->im = af::array(width, height, buffer).as(f32);
  gaf->im = af::transpose( gaf->im );
	gaf->im /= 255.0;

  if (bDoSetBackground) {
    gaf->im_background = af::array(width, height, buffer).as(f32);
    gaf->im_background= af::transpose( gaf->im_background );
    gaf->im_background/= 255.0;
    bDoSetBackground=0;
  }

  if (bDoSubtractBackground)
    gaf->im -= gaf->im_background;

  if (bDoReplaceSubtracted) {
    gaf->im_temp = gaf->im.copy();
    gaf->im_temp *= 255.0;
    gaf->im_temp = af::transpose(gaf->im_temp).as(u8);

    uint8_t *host = gaf->im_temp.host<uint8_t>();

    struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
    memcpy((void*)((char *)(shmem_region2.get_address())+height*width*nCurrRing), host,
           height*width);
  }

#if VERBOSE
  spdlog::info("mean={} min={} max={}",(float)af::mean<float>(im),
         (float)af::min<float>(im),
               (float)af::max<float>(im) );
#endif

	gaf->weighted_x = gaf->im * gaf->im_xcoor;
	gaf->weighted_y = gaf->im * gaf->im_ycoor;

#if VERBOSE
  spdlog::info("mean={} min={} max={}",(float)af::mean<float>(weighted_x),
               (float)af::min<float>(weighted_x),
               (float)af::max<float>(weighted_x) );
#endif

	int doprint=0;
	if (doprint)
		af_print(gaf->seq1);

	gaf->atemp = gaf->weighted_x(gaf->seq1); //weighted_x or im_xcoor for debug

#if VERBOSE
  spdlog::info("mean={} min={} max={}",(float)af::mean<float>(atemp),
               (float)af::min<float>(atemp),
               (float)af::max<float>(atemp) );
#endif

	if (doprint*0)
		af_print(gaf->atemp );
	gaf->x_reshape = moddims( gaf->atemp, gaf->new_dims );

	gaf->atemp = gaf->weighted_y(gaf->seq1);
	gaf->y_reshape = af::moddims( gaf->atemp, gaf->new_dims );
	if (doprint*0)
		af_print(gaf->y_reshape );

	gaf->im2 = gaf->im(gaf->seq1);
	gaf->im2 = af::moddims( gaf->im2, gaf->new_dims );

	gaf->sums = af::sum( gaf->im2, 0);
	gaf->sums_x = af::sum( gaf->x_reshape, 0) / gaf->sums;
	gaf->sums_y = af::sum( gaf->y_reshape, 0) / gaf->sums;

  float *host_x = gaf->sums_x.host<float>();
  float *host_y = gaf->sums_y.host<float>();

  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
  memcpy(pShmemBoxes->centroid_x, host_x, sizeof(float)*num_boxes);
  memcpy(pShmemBoxes->centroid_y, host_y, sizeof(float)*num_boxes);

#if 0
  if (pShmemBoxes->header_version & 2) //Follow
  {
    memcpy(pShmemBoxes->box_x, host_x, sizeof(float)*num_boxes);
    memcpy(pShmemBoxes->box_y, host_y, sizeof(float)*num_boxes);
  }
#endif //0

#if VERBOSE
  spdlog::info("Val0 x,y: {},{}",host_x[0],host_y[0]);
	spdlog::info("Count: {}", (float)af::count<float>(sums_x));
#endif

  af::freeHost(host_x);
  af::freeHost(host_y);

	if (0) {
		af_print( gaf->sums_x );
		af_print( gaf->sums_y );
	}

#if 0
	spdlog::info("{}", num_boxes);
	spdlog::info("Maxx: {}", (float)af::max<float>(gaf->sums_x));
	spdlog::info("Minx: {}", (float)af::min<float>(gaf->sums_x));
	spdlog::info("Mayy: {}", (float)af::max<float>(gaf->sums_y));
	spdlog::info("Miny: {}", (float)af::min<float>(gaf->sums_y));
	spdlog::info("Meanx: {}", (float)af::mean<float>(gaf->sums_x));
	spdlog::info("Meany: {}", (float)af::mean<float>(gaf->sums_y));

  spdlog::info("Val100 x,y: {},{}",host_x[2],host_y[2]);
#endif

  return 0;
}

void wait_for_lock(uint8_t *lock) {
  while (*lock) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  *lock=1;
}

void unlock(uint8_t *lock) {
  *lock=0;
}

PLUGIN_API(centroiding,process,char *params)
{
  char *msg=socket_check(CENTROIDING_SOCKET);
  if (msg!=NULL) {
    //spdlog::info("CEN: {}",msg);
    switch (msg[0]) {
    case 'B':
      spdlog::info("B!");
      bDoSubtractBackground=1;
      break;
    case 'b':
      spdlog::info("b!");
      bDoSubtractBackground=0;
      break;
    case 'S':
      spdlog::info("S!");
      bDoSetBackground=1;
      break;
    case 'R':
      spdlog::info("R!");
      bDoReplaceSubtracted=1;
      break;
    case 'r':
      spdlog::info("r!");
      bDoReplaceSubtracted=0;
      break;
    default:
      spdlog::error("CEN unknown");
    }
  }

  // TODO: is dynamically changing the size allowed?
	struct shmem_header* pShmem = (struct shmem_header*) shmem_region1.get_address();

	uint16_t nCurrRing = pShmem->current_frame;
	uint16_t height = pShmem->dimensions[0];
	uint16_t width = pShmem->dimensions[1];

  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
  wait_for_lock(&pShmemBoxes->lock);

	memcpy((void*)buffer,
         ((const char *)(shmem_region2.get_address()))+height*width*nCurrRing, height*width);

  if (params[0]=='I') {
    read_boxes(width);
    spdlog::info("Got boxes");
    spdlog::info("Centroiding Dims: {} {} {}", width, height, int(buffer[0])) ;
  }

  find_cendroids_af(buffer, width, height);

  unlock(&pShmemBoxes->lock);
	return 0;
};

PLUGIN_API(centroiding,plugin_close,char *params)
{
  std::cout <<"CEN close\n" ;
  return 0;
};

PLUGIN_API(centroiding,set_params,char *settings)
{
  std::cout <<"P1 set_params " << settings<< "\n";
  return 0;
};

PLUGIN_API(centroiding,get_info,char *which)
{
  std::cout <<"P1 get_info";
  return 0;
};



