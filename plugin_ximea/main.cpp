/*
  This is the reference example application code for XIMEA cameras.
  You can use it to simplify development of your camera application.
  
  Sample name: 
    xiAPI / Capture-10-images

  Description: 
    Open camera, capture 10 images while printing first pixel from each image.

  Workflow:
    1: Open camera
    2: Set parameters
    3: Start acquisition
    4: Each image captured - print dimensions and value of the first pixel
*/

#include <iostream>
#include <memory.h>
#if defined (_WIN32)
#include <xiApi.h>       // Windows
#include <Windows.h>
#else
#include <m3api/xiApi.h> // Linux, OSX
#endif

// For NextWave Plugin
#include "nextwave_plugin.hpp"
#pragma pack(push,1)
#include "memory_layout.h"
#pragma pack(pop) // restore previous setting

#if _WIN64
// Add this directory (right-click on project in solution explorer, etc.)
#include "boost/interprocess/windows_shared_memory.hpp"
#include "boost/interprocess/mapped_region.hpp"
#else
#endif
using namespace boost::interprocess;

// Check error macro. It executes function. Print and throw error if result is not OK.
#define CE(func) {XI_RETURN stat = (func); if (XI_OK!=stat) {printf("Error:%d returned from function:"#func"\n",stat);throw "Error";}}

DECL init(void)
{
#if _WIN64
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    mapped_region shmem_region{ shmem, read_write };
	
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    mapped_region shmem_region2{ shmem2, read_write };	
#else
	// TODO: Linux version
#endif
	struct shmem_header* pShmHeader = (struct shmem_header*) shmem_region.get_address();
	struct shmem_header* pShmBuffer = (struct shmem_header*) shmem_region2.get_address();

	pShmHeader->lock = (uint8_t)1; // Everyone keep out until we are done!

	// Don't need to write these each time:
	pShmHeader->header_version = (uint8_t)NW_HEADER_VERSION;
	//pShmHeader->dimensions[0] = (uint16_t)height; // don't yet know height/width
	//pShmHeader->dimensions[1] = (uint16_t)width;
	pShmHeader->dimensions[2] = (uint16_t)0;
	pShmHeader->dimensions[3] = (uint16_t)0;
	pShmHeader->datatype_code = (uint8_t)NW_UINT8;
	pShmHeader->max_frames = (uint8_t)NW_MAX_FRAMES;

	int nCurrRing=0;

	pShmHeader->current_frame = (uint8_t)nCurrRing;
	pShmHeader->timestamps[nCurrRing] = (uint8_t)0;



			
	HANDLE xiH = NULL;
	try {	
		printf("Opening first camera...\n");
		CE(xiOpenDevice(0, &xiH));

		printf("Setting exposure time to 10ms...\n");
		CE(xiSetParamInt(xiH, XI_PRM_EXPOSURE, 1000));

		// Note:
		// The default parameters of each camera might be different in different API versions
		// In order to ensure that your application will have camera in expected state,
		// please set all parameters expected by your application to required value.

		printf("Starting acquisition...\n");
		CE(xiStartAcquisition(xiH));

		XI_IMG image; // image buffer
		//memset(&image, 0, sizeof(image));
		image.size = sizeof(XI_IMG);

		while (1) //((GetKeyState('Q') !=  0) )
		{
			int height, width;
			CE(xiGetImage(xiH, 5000, &image)); // getting next image from the camera opened
					
            pShmHeader->lock = (uint8_t)1; // Keep out until we are done!

			height=(int)image.height; (int)width=image.width;	// TODO: needn't do this every time
			

			// Current frame:
			pShmHeader->current_frame = (uint8_t)nCurrRing;
			pShmHeader->statuses[nCurrRing] = (uint8_t)NW_STATUS_READ;
			pShmHeader->timestamps[nCurrRing] = (uint64_t)image.tsUSec;
			

#if 1
			memcpy( ((uint8_t *)(pShmBuffer)+height*width*nCurrRing),
				(void*)image.bp,
				height*width);
#endif //0
				

            pShmHeader->lock = (uint8_t)0; // Keep out until we are done!

			nCurrRing += 1;
			if (nCurrRing >= NW_MAX_FRAMES)
				nCurrRing = 0;
		}

		printf("Stopping acquisition...\n");
		xiStopAcquisition(xiH);
	}
	catch (const char*)
	{
	}
	xiCloseDevice(xiH);
	printf("Done\n");
#if defined (_WIN32)
	Sleep(2000);
#endif
	return 0;
}

DECL do_process(void)
{
  std::cout <<"P1 do_process\n" ;
  return 0;
};

DECL set_params(const char* settings_as_json)
{
  std::cout <<"P1 set_params " << settings_as_json << "\n";
  return 0;
};
 
DECL get_info(const char* which_info)
{
  std::cout <<"P1 get_info";
  return 0;
};