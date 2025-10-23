
#include <arrayfire.h>

#include <stdio.h>

#include <fstream>
#include <iostream>

//#include <crtdbg.h>

using namespace std;

#include <json.hpp>
using json=nlohmann::json;

#include "spdlog/spdlog.h"

#include "nextwave_plugin.cpp"

// UI Socket communication
#include <cstring>
#define CENTROIDING_SOCKET 50008
#include "socket.cpp"

#define VERBOSE 0

// Globals
#define BUF_SIZE (2048*2048*16)

#define CLIPVAL 0.95

unsigned char buffer[BUF_SIZE];

uint8_t omit_boxes[MAX_BOXES];

uint8_t bDoSubtractBackground=0;
uint8_t bDoSetBackground=0;
uint8_t bDoReplaceSubtracted=0;
uint8_t bDoThreshold=0;
uint8_t bBackgroundSet=0; // First time
uint8_t bUseMetric=0;

int16_t autoBack=-1;

float fThreshold=0.0;
float fGain=1.0;

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

void process_ui_commands(void); // Forward declr

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
  af::array im_subtracted_u8;

  af::array im_xcoor;
  af::array im_ycoor;

  af::array ref_x;
  af::array ref_y;

  af::array delta_x;
  af::array delta_y;
  af::array slopes;

  af::array weighted_x;
  af::array weighted_y;

  af::array sums_x;
  af::array sums_y;

  af::array seq1;

  af::dim4 new_dims;//(BOX_SIZE*BOX_SIZE,NBOXES); // AF stacks the boxes, so do this to reshape for easier sum reduction

  af::array refs_next;
  af::array ref_x_shift;
  af::array ref_y_shift;

  // These are "temp" & overwritten each time in the loop, but put here to avoid GC, etc. :
  af::array im2;
  af::array sums;

  af::array atemp;
  af::array x_reshape;
  af::array y_reshape;
  
  af::array submask;

 // af::array influence;
  af::array influence_inv;
  af::array mirror_voltages;
  
  af::array dummy; // To try to prevent crash
};

class af_instance *gaf;

float fbuffer[BUF_SIZE];
int nbuffer[BUF_SIZE];

uint16_t num_boxes;
double pixel_um;
double box_size_float;
double focal_length_um;
int box_size;

double pupil_radius_pixels;
double pupil_radius_um;

int width;
int height;

int nActuators;
int nTerms;

// Local buffer to copy from shmem into before af.
// Need to hold either box positions or inv. influence matrix
double local_buf[MAX_TERMS];

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

// TODO: use new scheme
struct shmem_header* pShmem; // = (struct shmem_header*) shmem_region1.get_address();

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

	if (!bBackgroundSet) {
		gaf->im_background = af::array(width, height, buffer).as(f64);
		gaf->im_background= af::transpose( gaf->im_background ) * 0;
		bBackgroundSet=1;
	};
}

#define EXTENT 5
// Submask will comprise up to (extent*2+1) on each side  ( -EXTENT to middle pixel to +EXTENT )

void rcv_boxes(int width) {

  // TODO: is dynamically changing the size allowed?
  struct shmem_header* pShmem = (struct shmem_header*) shmem_region1.get_address();
  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();

  num_boxes = pShmemBoxes->num_boxes;
  pixel_um = pShmemBoxes->pixel_um;
  pupil_radius_um = pShmemBoxes->pupil_radius_um;
  box_size_float = pShmemBoxes->box_um/pixel_um;
  focal_length_um = pShmemBoxes->focal_um;
  box_size = (int)(box_size_float+0.5); // round up to nearest integer

  pupil_radius_pixels = pupil_radius_um/pixel_um*1000.0;

  height = pShmem->dimensions[0];
  width = pShmem->dimensions[1];

  // TODO: this will be slow to do every time.
  init_buffers(width, height, box_size, num_boxes);

  // TODO: can this be done without the memcpy to local?
  memcpy(local_buf, pShmemBoxes->reference_x, sizeof(CALC_TYPE) * num_boxes);
  gaf->ref_x = af::array(1, num_boxes, local_buf);
  memcpy(local_buf, pShmemBoxes->reference_y, sizeof(CALC_TYPE) * num_boxes);
  gaf->ref_y = af::array(1, num_boxes, local_buf);

  gaf->ref_x_shift=gaf->ref_x * 0;
  gaf->ref_y_shift=gaf->ref_y * 0;

  spdlog::info("InIt: {}x{} #:{} {} {} {} {}\n", width,height, num_boxes, box_size, pShmemBoxes->box_x[0], pShmemBoxes->box_y[0]-box_size, pShmemBoxes->box_y[0]-box_size/2);

  // Each box will have a set of 1D indices into its members
	for (int ibox=0; ibox<num_boxes; ibox++) {
    //spdlog::info("{} {}\n", pShmemBoxes->box_x[ibox], pShmemBoxes->box_y[ibox]);
		int ibox_x = pShmemBoxes->box_x[ibox]-box_size/2;
		int ibox_y = pShmemBoxes->box_y[ibox]-box_size/2;
		for (int y=0; y<box_size; y++)  {
			for (int x=0; x<box_size; x++) {
        int posx=ibox_x+x;
        int posy=ibox_y+y;
        int idx_1d=posx*height+posy; // Column major // TODO: height
        if (ibox==0 && y<0) { // change y< for debugging
          printf("%d %d %d\n",idx_1d, posx, posy);
        }
				//nbuffer[ibox*box_size*box_size+x*box_size+y]=idx_1d;
				nbuffer[ibox*box_size*box_size+x*box_size+y]=idx_1d;
			}
      if (ibox==0 && y<0) { // change y< for debugging
        printf("\n");
      }
		}
	}
	gaf->seq1 = af::array(box_size*box_size, num_boxes, nbuffer );

	for (int ibox=0; ibox<box_size*box_size; ibox++) {
		int x_within = ibox / box_size ;
		int y_within = ibox % box_size;
		for (int y=0; y<box_size; y++)  {
			for (int x=0; x<box_size; x++) {
				int maskval=1;
				if ((x-x_within) > EXTENT)
					maskval=0;
				if ((x_within-x) > EXTENT)
					maskval=0;
				if ((y_within-y) > EXTENT)
					maskval=0;
				if ((y-y_within) > EXTENT)
					maskval=0;
				nbuffer[ibox*box_size*box_size+x*box_size+y]=maskval;
			}
		}
	}
 	gaf->submask = af::array(box_size*box_size, box_size*box_size, nbuffer );
 	gaf->submask = af::reorder(gaf->submask, 1, 0);
 	//gaf->submask = af::transpose(gaf->submask);

  spdlog::info("boxes: #={} size={} pupil={}", num_boxes, box_size, pupil_radius_pixels );
  //spdlog::info("x0={} y0={}", x_ref0, y_ref0 );

#if VERBOSE
  printf("%f\n",(float)af::max<float>(gaf->seq1) );
  printf("mn1: %f\n",(float)af::min<float>(gaf->seq1(af::span,1)) );
  printf("mn2: %f\n",(float)af::min<float>(gaf->seq1(af::span,2)) );
  printf("mx1: %f\n",(float)af::max<float>(gaf->seq1(af::span,1)) );
  printf("mx2: %f\n",(float)af::max<float>(gaf->seq1(af::span,2)) );
  printf("count1: %f\n",(float)af::count<float>(gaf->seq1(af::span,1)) );
#endif

	nActuators = pShmemBoxes->nActuators;
	nTerms = pShmemBoxes->nTerms; // should be (valid) boxes * 2

	//memcpy(local_buf, pShmemBoxes->influence, sizeof(CALC_TYPE) * nActuators*nTerms);
	//gaf->influence = af::array(nActuators, nTerms, local_buf );
	memcpy(local_buf, pShmemBoxes->influence_inv, sizeof(CALC_TYPE) * nActuators*nTerms);
	gaf->influence_inv = af::array(nTerms, nActuators, local_buf );

	spdlog::info("CEN rcv: {} {} {} {} {} {} {}x{} infl:{}x{} {}",
		num_boxes, pixel_um, box_size, box_size_float, pupil_radius_um, pupil_radius_pixels,
		width, height, nTerms, nActuators, (float)af::max<float>(gaf->influence_inv) );

}

PLUGIN_API(centroiding,init,char *params)
{
  try {
    af::setBackend(AF_BACKEND_CUDA);
    spdlog::info("Set AF_BACKEND_CUDA");
  } catch (...){
	  try {
		af::setBackend(AF_BACKEND_OPENCL);
		spdlog::info("Set AF_BACKEND_OPENCL");
	  } catch (...){
		  try {
			  af::setBackend(AF_BACKEND_CPU);
			  spdlog::info("Set AF_BACKEND_CPU");
		} catch (...){
		  spdlog::error("Couldn't load any AF_BACKEND!!");
		}
	  }
  }

  af::info();
  //af::deviceInfo();
	
  plugin_access_shmem();
  pShmem = (struct shmem_header*) shmem_region1.get_address();
  
  gaf=new af_instance();

  //process_ui_commands(); // First read/check opens the pipe that Python UI sockets needs

  //spdlog::info("Ok");

  return 0;
}

// https://stackoverflow.com/questions/49692531/transfer-data-from-arrayfire-arrays-to-armadillo-structures
// https://github.com/arrayfire/arrayfire/issues/1405;
int find_centroids_af(unsigned char *buffer, int width, int height) {
	
	// Image is a 64-bit float (0-1.0)
	gaf->im = af::array(width, height, buffer).as(f64);
	gaf->im = af::transpose( gaf->im );
	gaf->im /= 255.0;
	
    struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
	memcpy(omit_boxes, pShmemBoxes->centroid_omit, sizeof(pShmemBoxes->centroid_omit[0])*num_boxes);
	auto afOmits = af::array( num_boxes, omit_boxes).as(b8);
	
  if (bDoSetBackground) {
    gaf->im_background = af::array(width, height, buffer).as(f64);
    gaf->im_background= af::transpose( gaf->im_background );
    gaf->im_background/= 255.0;
    bDoSetBackground=0;
  }
  if (autoBack>=0) {
	// Setting a mean singleton for the entire image (e.g. mode of any spot image)
    gaf->im_background = af::array(width, height, buffer).as(f64);
    gaf->im_background = gaf->im_background * 0 + autoBack;
    gaf->im_background/= 255.0;
    autoBack = -1;
  }

  if (bDoSubtractBackground) {
    gaf->im -= gaf->im_background;
    gaf->im(gaf->im<0) = 0;
  }
  
  // Threshold pixels: (could also binarize )
  if (bDoThreshold) {
    //float thresholdValue = 60.0/255.0;
    gaf->im = (gaf->im < fThreshold) * 0 + gaf->im * (gaf->im >= fThreshold);
  }

  gaf->im_subtracted_u8 = gaf->im.copy();
  gaf->im_subtracted_u8 *= 255.0;
  gaf->im_subtracted_u8 = af::transpose(gaf->im_subtracted_u8).as(u8);

  // Sorry, need to zero out the zeroth entry, since we might use that with in the mask
  // to get rid of unwanted pixels.
  gaf->im(0) = 0;
  
  uint8_t *host_im_subtracted_u8 = gaf->im_subtracted_u8.host<uint8_t>();

  if (bDoReplaceSubtracted) {
    //struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
    memcpy((void*)((char *)(shmem_region2.get_address())+height*width*nCurrRing), host_im_subtracted_u8,
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
	if (0)
    af_print(gaf->seq1(af::span,0));


#if VERBOSE
  spdlog::info("mean={} min={} max={}",(float)af::mean<float>(atemp),
               (float)af::min<float>(atemp),
               (float)af::max<float>(atemp) );
#endif

	gaf->atemp = gaf->weighted_x(gaf->seq1); //weighted_x or im_xcoor for debug
	if (doprint*0)
		af_print(gaf->atemp );
	gaf->x_reshape = af::moddims( gaf->atemp, gaf->new_dims );

	gaf->atemp = gaf->weighted_y(gaf->seq1);
	gaf->y_reshape = af::moddims( gaf->atemp, gaf->new_dims );
	if (doprint*0)
		af_print(gaf->y_reshape );

	gaf->im2 = gaf->im(gaf->seq1);
	
	// Reshape the array so each box can be summed  
	gaf->im2 = af::moddims( gaf->im2, gaf->new_dims );
	
    af::array max_vals, max_indices, box_masks;
    af::max(max_vals, max_indices, gaf->im2, 0); // 0=within each box. Find max pixel in each box, put idxs into max_indices
	//max_indices = af::reorder(max_indices, 1, 0);
    //spdlog::info("mean={}",(float)af::mean<float>(max_indices) );

    //float idx0 = max_indices(0,0).scalar<float>();
    //spdlog::info("max0={}", idx0 );
    //af_print(max_indices(af::seq(2))); 
	
	// Submask for each box will comprise the appropriate mask for its Max pixel
	box_masks = gaf->submask(af::span, max_indices );
	
	// Disable MAX sub-boxing: (Make all full-mask masks.)
	//box_masks = box_masks * 0 + 1.0;

#if 0
	af::dim4 dimX = gaf->submask.dims();
	spdlog::info("{},{}",dimX[0], dimX[1]);
	dimX = box_masks.dims();
	spdlog::info("{},{}",dimX[0], dimX[1]);
	dimX = gaf->im2.dims();
	spdlog::info("{},{}",dimX[0], dimX[1]);
	dimX = max_indices.dims();
	spdlog::info("{},{}",dimX[0], dimX[1]);
	dimX = gaf->seq1.dims();
	spdlog::info("{},{}",dimX[0], dimX[1]);	
#endif //0	
	
	gaf->sums = af::sum( gaf->im2 * box_masks, 0);
	
	// Sum all pixels divide by total to get center of mass
	gaf->sums_x = af::sum( gaf->x_reshape * box_masks, 0) / gaf->sums;
	gaf->sums_y = af::sum( gaf->y_reshape * box_masks, 0) / gaf->sums;

  gaf->sums_x(afOmits) = af::NaN;
  gaf->sums_y(afOmits) = af::NaN; 


  // Compute deltas and write to shmem
  gaf->delta_x = gaf->sums_x - gaf->ref_x;
  gaf->delta_y = gaf->sums_y - gaf->ref_y;

  // TODO: Might this be slow?
  memcpy(gpShmemLog[gpShmemHeader->log_index].im, host_im_subtracted_u8, sizeof(uint8_t)*width*height);

  af::array metrics = gaf->sums / af::sum( gaf->im2, 0);

  // Indexes of box numbers that are okay ; // BOOL array
  //af::array valids = !af::isNaN (gaf->sums_x) && !af::isInf (gaf->sums_x) &&
  af::array valids2;

  af::array minimum_thresh = gaf->sums / ((EXTENT*2+1) * (EXTENT*2+1));
  af::array valids;
  if (bUseMetric)
	valids = (metrics > 0.5) && (minimum_thresh>0.04) && !af::isNaN (gaf->sums_x) && !af::isInf (gaf->sums_x); 
  else
	valids = (metrics > 0.0) && (minimum_thresh>0.04) && !af::isNaN (gaf->sums_x) && !af::isInf (gaf->sums_x); 

  af::array idx_valid=af::where(valids); // Array of non-zeros
  
  gaf->delta_x( af::where(!valids) ) = af::NaN;
  //uint8_t *host_valids = (af::mean( gaf->im2, 0)*255).as(u8).host<uint8_t>(); // To see the metric
  
  af::dim4 dims=idx_valid.dims();
  if (dims[0]==0 ) {
	spdlog::info("No valid. Skipping. {} {}",dims[0], dims[1]);
	goto free_afterwards;
  }
 // Check that valids are okay
 
  // Remove tip and tilt
  gaf->delta_x -= (CALC_TYPE)af::mean<CALC_TYPE>(gaf->delta_x(valids) ); // weighted mean, 0 for NaNs
  gaf->delta_y -= (CALC_TYPE)af::mean<CALC_TYPE>(gaf->delta_y(valids) );
  gaf->delta_y = -gaf->delta_y; // Negate y (coord system)

  //gaf->delta_x(nans) = 0.0;
  //gaf->delta_y(nans) = 0.0;

  
  // Interleave x and y slopes. Stack them side-by-side, transpose then flatten.
  // This is correct because they are arrays with dimensions (1,x)
  gaf->slopes = af::join(0, gaf->delta_x, gaf->delta_y); 
  gaf->slopes = af::moddims(gaf->slopes,af::dim4(1,num_boxes*2,1,1) ); // Flatten
  gaf->slopes /= (focal_length_um/pixel_um);

  valids2 = af::transpose( af::join(1, 2*idx_valid, 2*idx_valid+1 ) );
  valids2 = af::moddims(valids2,af::dim4(2*idx_valid.dims(0),1,1,1) ); // like flatten, but in 2nd dimension

  if (af::sum<int>(valids2)==0) {
	  gaf->mirror_voltages = gaf->influence_inv(0,af::span) * 0;
  } else {
	gaf->mirror_voltages = af::matmul(gaf->slopes(valids2), gaf->influence_inv(valids2,af::span) );
	//spdlog::info("{} {} {}",(float)af::max<float>(gaf->mirror_voltages), (float)af::max<float>(gaf->slopes(valids2)), idx_valid.dims(0) );
  }
  //gaf->mirror_voltages = af::matmul(gaf->slopes, (gaf->influence_inv) );
  

// DEBUGGING/logging

  if (0) {
  CALC_TYPE *host_delta_x = gaf->delta_x.host<CALC_TYPE>();
  CALC_TYPE *host_delta_y = gaf->delta_y.host<CALC_TYPE>();
  //memcpy(pShmemBoxes->delta_x, host_delta_x, sizeof(CALC_TYPE)*num_boxes);
  //memcpy(pShmemBoxes->delta_y, host_delta_y, sizeof(CALC_TYPE)*num_boxes);

  // If want to log deltas:
  memcpy(gpShmemLog[gpShmemHeader->log_index].centroid_x, host_delta_x, sizeof(CALC_TYPE)*num_boxes);
  memcpy(gpShmemLog[gpShmemHeader->log_index].centroid_y, host_delta_y, sizeof(CALC_TYPE)*num_boxes);
  }
  uint8_t *host_valids = (valids*100.0).as(u8).host<uint8_t>(); // To see the metric
  memcpy(pShmemBoxes->centroid_valid, host_valids, sizeof(uint8_t)*num_boxes); // 8 bits per box
  CALC_TYPE *host_x = gaf->sums_x.host<CALC_TYPE>();
  CALC_TYPE *host_y = gaf->sums_y.host<CALC_TYPE>();

  //af::array idx_double;
  CALC_TYPE *host_slopes = gaf->slopes.host<CALC_TYPE>(); // as(f64).
  memcpy(pShmemBoxes->box_x_normalized, host_slopes, sizeof(CALC_TYPE)*num_boxes*2); // Slopes have X and Y
  af::freeHost(host_slopes);
  
#if 0
  if (pShmemBoxes->header_version & 2) //Follow // TODO
  {
    memcpy(pShmemBoxes->box_x, host_x, sizeof(float)*num_boxes);
    memcpy(pShmemBoxes->box_y, host_y, sizeof(float)*num_boxes);
  }
#endif //0
  
  memcpy(pShmemBoxes->centroid_x, host_x, sizeof(CALC_TYPE)*num_boxes);
  memcpy(pShmemBoxes->centroid_y, host_y, sizeof(CALC_TYPE)*num_boxes);

  //memcpy(pShmemBoxes->mirror_voltages, host_mirror_voltages, sizeof(double)*nActuators); // Added for debugging 2025/26/02 -- see realtime voltage calcs

  double *host_mirror_voltages = gaf->mirror_voltages.host<double>();
  auto save1 = pShmemBoxes->mirror_voltages[0]; // Debuggi
  
	if ((pShmem->mode & MODE_AO) ) { // Closed loop
	
		double mirror_min=10, mirror_max=-10, mirror_mean=0;
		for (int i=0; i<nActuators; i++) {

			if (pShmem->mode & MODE_AO ) {
			  // If closed loop, add new voltages to old
			  pShmemBoxes->mirror_voltages[i] = pShmemBoxes->mirror_voltages[i] - host_mirror_voltages[i]*fGain;
			}
			// CLAMP
			if (pShmemBoxes->mirror_voltages[i] > CLIPVAL)
				pShmemBoxes->mirror_voltages[i]=CLIPVAL;
			if (pShmemBoxes->mirror_voltages[i] < -CLIPVAL)
				pShmemBoxes->mirror_voltages[i]=-CLIPVAL;

			if (!std::isfinite(pShmemBoxes->mirror_voltages[i]) ) {
				pShmemBoxes->mirror_voltages[i]=0; // ? flatten to zero if nan/inf
			}

			// Just for statistics
			if (pShmemBoxes->mirror_voltages[i] > mirror_max)
				mirror_max=pShmemBoxes->mirror_voltages[i];
			if (pShmemBoxes->mirror_voltages[i] < mirror_min)
				mirror_min=pShmemBoxes->mirror_voltages[i];

			mirror_mean += pShmemBoxes->mirror_voltages[i];
		}
		mirror_mean /= nActuators;

		// Memory Log: for debugging/log
		memcpy(gpShmemLog[gpShmemHeader->log_index].mirrors, pShmemBoxes->mirror_voltages, sizeof(pShmemBoxes->mirror_voltages[0])*nActuators);
		
		// Debug:
		//spdlog::info("Mirror {}:{}/{} 0:{} 00:{}", (double)mirror_mean, (double)mirror_min, (double) mirror_max, (double)pShmemBoxes->mirror_voltages[0], (double)save1 );

	} // if closed loop
  af::freeHost(host_mirror_voltages);
  af::freeHost(host_x);
  af::freeHost(host_y);

free_afterwards:
  af::freeHost(host_im_subtracted_u8);
  
  //spdlog::info("ZZZ");
  
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

void process_ui_commands(void) {
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
	case 's':
      autoBack = atoi(msg+2);
      spdlog::info("s={}",autoBack);
      break;
	
    case 'M':
      bUseMetric=1;
      break;
	case 'm':
      bUseMetric=0;
      break;
	
    case 'R':
      spdlog::info("R!");
      bDoReplaceSubtracted=1;
      break;
    case 'r':
      spdlog::info("r!");
      bDoReplaceSubtracted=0;
      break;
    case 'T':
      fThreshold=atof(msg+1);
      //spdlog::info("T={}",fThreshold);
      bDoThreshold=1;
      break;
    case 'G':
      fGain=atof(msg+1);
      spdlog::info("G={}",fGain);
      break;
    case 'E': {
      int xpos=atoi(msg+1);
      int ypos=atoi(msg+6);
	  spdlog::info("x,y={} {}",xpos,ypos);
	  gaf->im_background( af::seq(ypos, ypos+10), af::seq(xpos,xpos+10) ) = 1.0; // Totally hot
	  break;
	}
    case 't':
      spdlog::info("t!");
      bDoThreshold=0;
      break;
    default:
      spdlog::error("CEN unknown");
    }
  }
}

PLUGIN_API(centroiding,process,char *params)
{
  process_ui_commands();

	uint16_t nCurrRing = pShmem->current_frame;
	uint16_t height = pShmem->dimensions[0];
	uint16_t width = pShmem->dimensions[1];

  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
  //wait_for_lock(&pShmemBoxes->lock);

	memcpy((void*)buffer,
         ((const char *)(shmem_region2.get_address()))+height*width*nCurrRing, height*width);

  //double total=0;
  //for (int n=0; n<1000; n++)
	//  total += buffer[n]; 
  //spdlog::info("Centroiding_process ring: {} {}",nCurrRing,total/1000.0);

  if (params[0]=='I') {
    rcv_boxes(width);
    spdlog::info("Got boxes");
    spdlog::info("Centroiding Dims: {}x{} pixel0:{}", width, height, int(buffer[0])) ;
  }

  find_centroids_af(buffer, width, height);

  //unlock(&pShmemBoxes->lock);
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



// ArrayFire/debugging functions, here in case needed:
#if 0
  //spdlog::info(yuck.dims(0));

  
//int *host_valids = valids2.host<int>();  
  //spdlog::info("{} {} {} {} {} {} ",host_valids[0], host_valids[1], host_valids[2], host_valids[3], host_valids[4], host_valids[5]) ;//, host_valids[1], host_valids[2], host_valids[3] );
 //af::freeHost(host_valids);
 //af::print("valids2",valids2) ; //(1,af::seq(4)) );
  spdlog::info("M");

/*   spdlog::info( (float)af::sum<float>(valids) );

  spdlog::info((int)gaf->influence_inv.dims(0) );
  spdlog::info((int)gaf->influence_inv.dims(1) );
  spdlog::info((int)gaf->slopes.dims(0) );
  spdlog::info((int)gaf->slopes.dims(1) );
  */ //spdlog::info("N ",gaf->slopes(valids2).dims(0) ,gaf->slopes(valids2).dims(1), gaf->influence_inv.dims(0), gaf->influence_inv.dims(1) );

  //gaf->mirror_voltages = gaf->influence_inv(0,af::span);
  //gaf->mirror_voltages = af::constant<double>( 0, af::dim4(97,1,1,1) );
#endif //0: debugging functions
