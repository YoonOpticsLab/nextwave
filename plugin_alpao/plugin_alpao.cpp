#include <stdio.h>

#include <fstream>
#include <iostream>

//using namespace std;
#include "nextwave_plugin.hpp"
#include "nextwave_plugin.cpp"

// UI Socket communication
#include <cstring>
#define ALPAO_SOCKET 50010
#include "socket.cpp"

#define VERBOSE 0

#include "asdkDM.h"

acs::DM* dm;
int num_act;
acs::Scalar *data;

int mode=-1;
int loop=0;

#define REAL_AO 1

#define CLAMP_VAL 0.7

//PLUGIN_API(alpao,init,char *params)
DECL init(void)
{
  spdlog::info("ALPAO DM1");

  plugin_access_shmem();

  spdlog::info("ALPAO DM2");

#if REAL_AO  
  dm = new acs::DM( "BAX492" );
  num_act = (int)dm->Get( "NbOfActuator" );
#else
  num_act=97;
#endif
  spdlog::info("ALPAO DM ok: {}",num_act);

  data = new acs::Scalar[num_act]; // TODO: update if num_act changes (smaller pupil, etc.)

  for( int i=0; i<num_act; i++) {
	  data[i] = (acs::Scalar)0;
  }

#if REAL_AO
	dm->Send(data);
#endif
	  
  return 0;
}

DECL process(char *commands)
//PLUGIN_API(alpao,process,char *params)
{
  char *msg=socket_check(ALPAO_SOCKET);
  //if (msg!=NULL) {
  //  spdlog::info("ALPAO: {}",msg);
  //}
  if (commands[0]=='I')
	  if (commands[1]=='C')
		  loop=1;
	  else
		  loop=0;

  // TODO: is dynamically changing the size allowed?
	uint16_t nCurrRing = gpShmemHeader->current_frame;
	uint16_t height = gpShmemHeader->dimensions[0];
	uint16_t width = gpShmemHeader->dimensions[1];

   //wait_for_lock(&pShmemBoxes->lock);

  float mymin=10, mymax=-10, mymean=0;
  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();

  double val;
  for( int i=0; i<num_act; i++) {
	  val= pShmemBoxes->mirror_voltages[i];
	  
	  if (val<mymin) mymin=val;
	  if (val>mymax) mymax=val;
	  mymean += val;

	  // CLAMP
	  if (val > CLAMP_VAL)
		val=CLAMP_VAL;
	  if (val < -CLAMP_VAL)
		val=-CLAMP_VAL;
	  if (!std::isfinite(val) )
		val=0; // ?


	  data[i] = (acs::Scalar)val;	  
  };
  
  mymean /= num_act;
  
  spdlog::info("DM Loop:{} #{}:{}x{} {}{} min:{:0.4f} max:{:0.4f}, mean:{:0.4f} 0:{:0.4f} 1:{:0.4f} {}", loop, nCurrRing, height, width,
		commands[0], commands[1], mymin, mymax, mymean, data[0], data[1], val );
  
  if (loop) {
#if REAL_AO	 
	// Copied from documentation in ALPAO SDK3 Programmer's Guide  (pg 13/18)
	// Not sure why/if Stop() is needed.. (drc 06/2024)
	try
	{
	 dm->Send( data );
	 dm->Stop();
	}
	catch (std::exception e)
	{
		spdlog::error("ALPAO DM error: {}",e.what() );
	}
#endif
  };
  
  //unlock(&pShmemBoxes->lock);
  return 0;
};

DECL plugin_close(char *params)
//PLUGIN_API(alpao,plugin_close,char *params)
{
  std::cout <<"ALPAO close\n" ;
  
  //TODO: Zero mirror (?)
  delete [] data;
#if REAL_AO
  delete dm;
#endif  
  return 0;
};

PLUGIN_API(alpao,set_params,char *settings)
{
  std::cout <<"ALPAO set_params " << settings<< "\n";
  return 0;
};

PLUGIN_API(alpao,get_info,char *which)
{
  std::cout <<"ALPAO get_info";
  return 0;
};



