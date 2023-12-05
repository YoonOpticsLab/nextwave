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

// For timing stuff:
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

#include <json.hpp>
using json=nlohmann::ordered_json;

#include "memory_layout.h"

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

#if _WIN64
  name += std::string(".dll");
  handle=LoadLibrary(name.c_str());
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
  chkerr();
  //std::cout << handle << " Ok1\n" << std::flush;;

  int (*fptr )(const char*);
  int (*fptr2)(const char*);
  int (*fptr3)(const char*);
#if _WIN64
  *(int **)(&fptr)=(int *)GetProcAddress((HMODULE)handle,"init");
  *(int **)(&fptr2)=(int *)GetProcAddress((HMODULE)handle,"process");
  *(int **)(&fptr3)=(int *)GetProcAddress((HMODULE)handle,"plugin_close");
#else
  *(int **)(&fptr) = (int *)dlsym(handle,"init");
  *(int **)(&fptr2) = (int *)dlsym(handle,"process");
  *(int **)(&fptr3) = (int *)dlsym(handle,"plugin_close");
#endif
 
  aModule.handle=handle;
  aModule.name = name;
  aModule.fp_init=fptr;
  aModule.fp_do_process=fptr2;
  aModule.fp_close=fptr3;

chkerr();
//std::cout <<" Ok2\n" << std::flush;;

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

  std::list<struct module> listModules;

  std::ifstream fil(filename_config);
  json jdata = json::parse(fil);
	//std::cout << jdata.dump(2) << std::endl;
   //return 0;

  // iterate the array
  for (json::iterator it = jdata.begin(); it != jdata.end(); ++it) {
    //json jd2 = it.value(); // How to recurse
    //std::cout << jd2["data"] << "\n";

    std::string name=it.key();
    std::string value=to_string(it.value());

    spdlog::info("{} About to load", name);

    int err=load_module(name, value, listModules);
    spdlog::info("{} Loaded OK", name);
  }

  using namespace boost::interprocess;
#if _WIN64
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    windows_shared_memory shmem3(open_or_create, SHMEM_BUFFER_NAME2, read_write, (size_t)SHMEM_BUFFER_SIZE2);
#else

    struct shm_remove
    {
      shm_remove() { shared_memory_object::remove(SHMEM_HEADER_NAME); shared_memory_object::remove(SHMEM_BUFFER_NAME); shared_memory_object::remove(SHMEM_BUFFER_NAME2); spdlog::info("Ok. removed"); }
      ~shm_remove(){ shared_memory_object::remove(SHMEM_HEADER_NAME); shared_memory_object::remove(SHMEM_BUFFER_NAME); shared_memory_object::remove(SHMEM_BUFFER_NAME2); spdlog::info("Removed"); }
    } remover;

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

	spdlog::info("Before Q");
	spdlog::info("{}", listModules.size());
	char ps_nothing[64] = "test"; //.c_str();

#define REPS 10

  double ns[REPS*2]; //TODO

//	while ( (GetKeyState('Q') & 0x8000) == 0)
	for (int pipeline_count=0; pipeline_count<REPS; pipeline_count++)
	{
    int modnum=0;
		for (struct module it: listModules) {
			//int result=(*it.fp_init)(ps_nothing);
			//spdlog::info("About to run {}",it.name);
      high_resolution_clock::time_point time_before = high_resolution_clock::now();
			int result=(*it.fp_do_process)(ps_nothing);
      high_resolution_clock::time_point time_after = high_resolution_clock::now();

      duration<double> time_span = duration_cast<duration<double>>(time_after-time_before);
      ns[pipeline_count*2+modnum] = time_span.count();

      modnum++;
		}
#if 0
		if ( (GetKeyState('P') & 0x8000) != 0 )
		{
			spdlog::info("paused");

			while ( (GetKeyState('C') & 0x8000) == 0)
			{
				Sleep(250); // sleep for X ms
			};

			spdlog::info("continue");
		}
#endif //0

	};

	spdlog::info("After OK");
	for (int pipeline_count=0; pipeline_count<REPS; pipeline_count++)
    {
      int modnum=0;
      for (struct module it: listModules) {
        spdlog::info(ns[pipeline_count*2+modnum]);
        modnum++;
      }
    }

  // iterate the array
  for (struct module it: listModules) {
	char ps_nothing[64] = "test"; //.c_str();
	int result=(*it.fp_close)(ps_nothing);
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

