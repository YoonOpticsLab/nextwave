// # List of plugins
// #Processing_chain:
//- Camera -> centroids -> Zernikes -> AO


// #During each plugin, only change parameters when not processing. (i.e., between frames)

//#Closed loop

// https://stackoverflow.com/questions/11741580/dlopen-loadlibrary-on-same-application

#if 0
#include <thread>

void thread_entry(int foo, int bar)
{
  int result = foo + bar;
  // Do something with that, I guess
}


// Elsewhere in some part of the galaxy
std::thread thread(thread_entry, 5, 10);
// And probably
thread.detach();

// Or
std::thread(thread_entry).detach();
#endif //0

#include <iostream>
#include <fstream>
#include <list>

#include <chrono>
#include <thread>

#if _WIN64
  #include <windows.h>
  #include <strsafe.h>
  #include "boost/interprocess/windows_shared_memory.hpp"
  
  #define PLUGIN_API(prefix,which,params) extern "C"  __declspec(dllexport) int prefix##_##which(params)
  PLUGIN_API(centroiding,process,char *params); // TODO: test
#else
  #include <boost/interprocess/shared_memory_object.hpp>
  // On unix, for dl* functions:
  #include <dlfcn.h>
#endif
#include <boost/interprocess/mapped_region.hpp>

// For timing stuff:
#define BOOST_CHRONO_HEADER_ONLY
#include "boost/chrono.hpp"
using namespace std::chrono;

#include "spdlog/spdlog.h"

// Hints: https://stackoverflow.com/questions/11741580/dlopen-loadlibrary-on-same-application

#if _WIN64
void ErrorExit(LPTSTR lpszFunction) 
{ 
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError(); 

    if (dw==0) {
      return;
    }

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
        (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR)); 
    StringCchPrintf((LPTSTR)lpDisplayBuf, 
        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        TEXT("%s failed with error %d: %s"), 
        lpszFunction, dw, lpMsgBuf); 
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK); 

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    //ExitProcess(dw); 
}
#endif

void chkerr(void)
{
#if _WIN64
	ErrorExit("");
#else
  char *errstr = dlerror();
  if (errstr != NULL) {
    std::cout << "Error: " << errstr << "\n" << std::flush;
  }
#endif
}

#define PLUGIN_API(prefix,which,params) int prefix##which(params)
PLUGIN_API(centroiding,init,char *params);

#include <json.hpp>
using json=nlohmann::ordered_json;

#pragma pack(push,1)
#include "memory_layout.h"
#include "layout_boxes.h"
#include "layout_log.h"
#pragma pack(pop) // restore previous setting

struct module {
public:
  std::string name;
  // TODO: port # for pipe comm
  void *handle; // Does this work for Windows also?
  int (*fp_init)(const char*);
  int (*fp_close)(const char*);
  int (*fp_do_process)(const char*);
  int (*fp_set_params)(const char*);
  int (*fp_get_params)(const char*);
};

int load_module(std::string name, std::string params, std::list<struct module> &listModules)
{
#if _WIN64
  void *handle=0;
#else
  void *handle=0;
#endif

  struct module aModule;

  // If name does NOT begin with plugin, bail.
  if (name.find("plugin") != 0) {
    return -1;
  };

  spdlog::info("Name: {}", name);
#if _WIN64
  if (name=="plugin_centroiding") {
    spdlog::info("Local exe override");
    // Centroiding need to live in the EXE (I think linking with the AF DLL problematic)
    handle=GetModuleHandle(NULL);
  } else {
    name += std::string(".dll");
    handle=LoadLibrary(name.c_str());
  }
#else
  name = std::string("./lib") + name + std::string(".so");
  handle=dlopen(name.c_str(), RTLD_NOW|RTLD_LOCAL);
#endif

  if (handle==0) {
    spdlog::error("Couldn't load '{}'\n",name);
    chkerr();
    return -1;
  };

  // Error check. What to do if can't load?
  //chkerr();

  int (*fptr1)(const char*);
  int (*fptr2)(const char*);
  int (*fptr3)(const char*);

#if _WIN64
  std::string prefix="";
  if (name=="plugin_centroiding") {
    prefix="centroiding_";
  }
  std::string name1=prefix+"init";
  spdlog::info("trying: {}", name1 );
  *(int **)(&fptr1)=(int *)GetProcAddress((HMODULE)handle,name1.c_str());
  std::string name2=prefix+"process";
  *(int **)(&fptr2)=(int *)GetProcAddress((HMODULE)handle,name2.c_str());
  std::string name3=prefix+"plugin_close";
  *(int **)(&fptr3)=(int *)GetProcAddress((HMODULE)handle,name3.c_str());
#else
  *(int **)(&fptr1) = (int *)dlsym(handle,"init");
  *(int **)(&fptr2) = (int *)dlsym(handle,"process");
  *(int **)(&fptr3) = (int *)dlsym(handle,"plugin_close");
#endif
 
  aModule.handle=handle;
  aModule.name = name;
  aModule.fp_init=fptr1;
  aModule.fp_do_process=fptr2;
  aModule.fp_close=fptr3;

chkerr();

 spdlog::info("Loaded {} {}", name, handle);
 int result=(**aModule.fp_init)(params.c_str());
 spdlog::info("Inited {} {}", name, handle);

 listModules.push_back(aModule);
 return 0; // OK
}

using std::to_string;

int main(int argc, char** argv)
{
  if (argc<2) {
    std::cout << "Please specify a config file." << "\n";
    exit(-1);
  };
  std::string filename_config = argv[1];

  spdlog::info("Main");

  using namespace boost::interprocess;
#if _WIN64
    windows_shared_memory shmem1(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    windows_shared_memory shmem3(open_or_create, SHMEM_BUFFER_NAME_BOXES, read_write, (size_t)sizeof(shmem_boxes_header));
    windows_shared_memory shmem4(open_or_create, SHMEM_LOG_NAME, read_write, (size_t)(sizeof(shmem_log_entry)*SHMEM_LOG_MAX));
#else
    struct shm_remove
    {
      shm_remove() {
        shared_memory_object::remove(SHMEM_HEADER_NAME);
        shared_memory_object::remove(SHMEM_BUFFER_NAME);
        shared_memory_object::remove(SHMEM_BUFFER_NAME_BOXES);
        shared_memory_object::remove(SHMEM_LOG_NAME);
        spdlog::info("Ok. removed");
      }
      ~shm_remove() {
        shared_memory_object::remove(SHMEM_HEADER_NAME);
        shared_memory_object::remove(SHMEM_BUFFER_NAME);
        shared_memory_object::remove(SHMEM_BUFFER_NAME_BOXES);
        shared_memory_object::remove(SHMEM_LOG_NAME);
        spdlog::info("Removed"); }
    } remover;

    shared_memory_object shmem1(open_or_create, SHMEM_HEADER_NAME, read_write);
    shmem1.truncate((size_t)SHMEM_HEADER_SIZE);
    shared_memory_object shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write);
    shmem2.truncate((size_t)SHMEM_BUFFER_SIZE);
    shared_memory_object shmem3(open_or_create, SHMEM_BUFFER_NAME_BOXES, read_write);
    shmem3.truncate((size_t)sizeof(shmem_boxes_header));
    shared_memory_object shmem4(open_or_create, SHMEM_LOG_NAME, read_write);
    shmem4.truncate((size_t)(sizeof(shmem_log_entry)*SHMEM_LOG_MAX));
    //spdlog::info("Siz: {}",sizeof(shmem_boxes_header) );
#endif

    // Common to both OSes:
    mapped_region shmem_region1{ shmem1, read_write };
    mapped_region shmem_region2{ shmem2, read_write };
    mapped_region shmem_region3{ shmem3, read_write };
    mapped_region shmem_region4{ shmem4, read_write };

  // Read config file with modules
  std::list<struct module> listModules;
  std::ifstream fil(filename_config);
  json jdata = json::parse(fil);
  for (json::iterator it = jdata.begin(); it != jdata.end(); ++it) {

    std::string name=it.key();
    std::string value=to_string(it.value());

    spdlog::info("{} About to load", name);

    int err=load_module(name, value, listModules);
    spdlog::info("{} Loaded OK", name);
  }
	spdlog::info("Before Q");
	spdlog::info("{} modules", listModules.size());

  struct shmem_header* pShmem1 = (struct shmem_header*) shmem_region1.get_address();
  struct shmem_boxes_header* pShmemBoxes = (struct shmem_boxes_header*) shmem_region3.get_address();
  struct shmem_log_entry* pShmemLog = (struct shmem_log_entry*) shmem_region4.get_address();
 
  // Clear this before start anything since use as a sentinel
  pShmemBoxes->num_boxes=0;

#define REP_LOG 20

#define CLOCK_MULT 1e5
  typedef boost::chrono::duration<long long, boost::micro> microseconds_type;
  typedef boost::chrono::seconds mst;

  while ( pShmem1->mode==MODE_OFF || pShmemBoxes->num_boxes==0 ) {
    // sleep until the UI is ready to tell us to do something
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // TODO: Call proper init here somehow?

  double times_total[10]; //TODO

  uint16_t ns[REP_LOG*4]; //TODO
  char str_message[64] = "    "; //.c_str();
  long int pipeline_count=0;
  uint8_t bRunning=0;
  boost::chrono::high_resolution_clock::time_point time_start = boost::chrono::high_resolution_clock::now();

  while ( pShmem1->mode!=MODE_QUIT ) {

    if (pShmem1->mode > MODE_READY ) {

      if (pShmem1->mode == MODE_RUNONCE_CENTROIDING_AO) {
        str_message[0]='I'; // Re-init on "snap"
        str_message[1]='C'; // Re-init on "snap"
        bRunning=0;
      } else if (pShmem1->mode == MODE_RUNONCE_CENTROIDING) {
        str_message[0]='I'; // Re-init on "snap"
        str_message[1]=' '; // Re-init on "snap"
        bRunning=0;
      } else {
        str_message[0]=' ';
        str_message[1]=' ';

        if (!bRunning) { // If not running before, i.e. (re)-started
          pShmem1->total_frames=0;
          bRunning=1;
          time_start = boost::chrono::high_resolution_clock::now();
        }
      }

      boost::chrono::high_resolution_clock::time_point time_total_before = boost::chrono::high_resolution_clock::now();
      boost::chrono::high_resolution_clock::time_point time_start_frame = boost::chrono::high_resolution_clock::now();

      int modnum=0;
      int logidx=pipeline_count % REP_LOG;

      uint16_t times_local[4]; //TODO
      double dur;
      uint16_t ms_times_100;
      double times_before[4]; //TODO
      for (struct module it: listModules) {
          boost::chrono::high_resolution_clock::time_point time_before = boost::chrono::high_resolution_clock::now();

          int result=(*it.fp_do_process)((const char*)str_message);

          boost::chrono::high_resolution_clock::time_point time_after = boost::chrono::high_resolution_clock::now();
          boost::chrono::duration<double>micros = time_after - time_before;

          dur = micros.count();
          ms_times_100 = (uint16_t)(dur*CLOCK_MULT);

          ns[logidx*3+modnum] = ms_times_100;
          times_local[modnum] = ms_times_100;

          boost::chrono::duration<double>time_since_start = time_before - time_start;
          times_before[modnum] = time_since_start.count();

          modnum++;
        }

        pShmemLog[pShmem1->total_frames].frame_number=pShmem1->current_frame;
        pShmemLog[pShmem1->total_frames].total_frame_number=pShmem1->total_frames;
        pShmemLog[pShmem1->total_frames].time0=times_before[0];
        pShmemLog[pShmem1->total_frames].time1=times_before[1];
        pShmemLog[pShmem1->total_frames].time2=times_before[2];

        boost::chrono::high_resolution_clock::time_point time_now = boost::chrono::high_resolution_clock::now();
        //boost::chrono::duration<double> time_span = duration_cast<duration<double>>(time_now - time_total_before);
        boost::chrono::duration<double> time_span = time_now - time_total_before;
        times_total[pipeline_count % 10] = time_span.count();
        pShmem1->fps[0]=(uint16_t)(CLOCK_MULT*time_span.count()); // TODO: take mean

        pShmem1->fps[1]=times_local[0];
        pShmem1->fps[2]=times_local[1];
        //pShmem1->fps[1]=(uint16_t)(ns[logidx*2]);
        //pShmem1->fps[2]=(uint16_t)(ns[logidx*2+1]);

        pShmem1->total_frames++;

        if ( pShmem1->header_version==99 || pShmem1->mode==MODE_QUIT ) {
          break;
        };

        pipeline_count += 1;

      // Ran once, unarm
        if (pShmem1->mode & MODE_RUNONCE_CENTROIDING ) {
          pShmem1->mode = MODE_READY;
        } else {
		
			// TODO: Some kind of problems in closed loop... going too fast? 
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}


    } else { // Not running at all. Sleep a bit until summoned.
      bRunning=0;	  
	  
	  if (pShmem1->mode == MODE_OFF)
		pShmem1->total_frames=0; // TODO: not sure if we want this long-term
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

  }

	spdlog::info("Quit");
	for (int pipeline_count=0; pipeline_count<REP_LOG; pipeline_count++)
    {
      int modnum=0;
      for (struct module it: listModules) {
        spdlog::info("{}",ns[pipeline_count*2+modnum]);
        modnum++;
      }
    }

  // iterate the array
  for (struct module it: listModules) {
	char str_nothing[64] = ""; //.c_str();
	spdlog::info("Closing {}", it.name);
	int result=(*it.fp_close)(str_nothing);
	#if _WIN64
		FreeLibrary( (HMODULE)it.handle);
	#else
		dlclose(it.handle);
	#endif

		std::string name = it.name;
		spdlog::info("Freed {}", name);
	}

  chkerr();
}

