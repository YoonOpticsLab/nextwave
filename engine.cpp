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

#include "spdlog/spdlog.h"

#include <boost/interprocess/mapped_region.hpp>

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
using json=nlohmann::json;

#include "memory_layout.h"

struct module {
public:
  std::string name;
  // TODO: port # for pipe comm
  void *handle; // Does this work for Windows also?
  int (*fp_init)(const char*);
  int (*fp_do_process)(const char*);
  int (*fp_set_params)(const char*);
  int (*fp_get_params)(const char*);
};

int load_module(std::string name, std::string params, std::list<struct module> listModules)
{
#if _WIN64
  void *handle=0;
#else
  void *handle=0;
#endif

  struct module aModule;

#if _WIN64
  handle=LoadLibrary(name.c_str());
#else
  handle=dlopen(name.c_str(), RTLD_NOW|RTLD_LOCAL);
#endif

  if (handle==0) {
    printf("Couldn't load: Handle==0");
    chkerr();
    return -1;
  };

  // Error check. What to do if can't load?
  chkerr();
  //std::cout << handle << " Ok1\n" << std::flush;;

  int (*fptr)(const char*);
#if _WIN64
  *(int **)(&fptr)=(int *)GetProcAddress((HMODULE)handle,"init");
#else
  *(int **)(&fptr) = (int *)dlsym(handle,"init");
#endif
  aModule.fp_init=fptr;

chkerr();
//std::cout <<" Ok2\n" << std::flush;;

 spdlog::info("Loaded {} {}", name, handle);
 int result=(*fptr)(params.c_str());
 spdlog::info("Inited {} {}", name, handle);

 aModule.handle=handle;
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

  // iterate the array
  for (json::iterator it = jdata.begin(); it != jdata.end(); ++it) {
    //json jd2 = it.value(); // How to recurse
    //std::cout << jd2["data"] << "\n";

    std::string name=it.key();
    std::string value=to_string(it.value());

    //spdlog::info("Once {}", value);
    //std::cout << name << ':' << value << "\n";

    int err=load_module(name, value, listModules);
  }

  using namespace boost::interprocess;
#if _WIN64
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
#else

    struct shm_remove
    {
      shm_remove() { shared_memory_object::remove(SHMEM_HEADER_NAME); }
      ~shm_remove(){ shared_memory_object::remove(SHMEM_BUFFER_NAME); spdlog::info("Removed"); }
    } remover;

 shared_memory_object shmem(open_or_create, SHMEM_HEADER_NAME, read_write);
    shmem.truncate((size_t)SHMEM_HEADER_SIZE);
    shared_memory_object shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write);
    shmem2.truncate((size_t)SHMEM_BUFFER_SIZE);
#endif

    // Common to both OSes:
    mapped_region shmem_region{ shmem, read_write };
    mapped_region shmem_region2{ shmem2, read_write };
  //spdlog::info("Main {}", jdata["pi"]);

  // range-based for
  //for (auto& element : jdata) {
  //std::cout << element << '\n';
    //std::cout << element.key() << ':' << element.value();
  //}

//const char *text="Hello";
  //#if _WIN64
  //val=(*fptr)();
  ////#else
  //val=(*fptr)();
  //#endif

  //std::cout <<" Ok3\n" << std::flush;;
  //std::cout << val;

  //#if _WIN64
  //#else
  //dlclose(handle);
  //#endif

  chkerr();
}

