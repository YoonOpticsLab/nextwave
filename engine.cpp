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

#if _WIN64
#include <windows.h>
#include <strsafe.h>

#include "boost/interprocess/windows_shared_memory.hpp"
#include "boost/interprocess/mapped_region.hpp"

#else
// On unix, for dl* functions:
#include <dlfcn.h>
#endif
// Hints: https://stackoverflow.com/questions/11741580/dlopen-loadlibrary-on-same-application


void ErrorExit(LPTSTR lpszFunction) 
{ 
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError(); 

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

#include "memory_layout.h"

using namespace boost::interprocess;

int main()
{
#if _WIN64
    windows_shared_memory shmem(open_or_create, SHMEM_HEADER_NAME, read_write, (size_t)SHMEM_HEADER_SIZE);
    mapped_region shmem_region{ shmem, read_write };
	
    windows_shared_memory shmem2(open_or_create, SHMEM_BUFFER_NAME, read_write, (size_t)SHMEM_BUFFER_SIZE);
    mapped_region shmem_region2{ shmem2, read_write };	
#else
	// TODO: Linux version
#endif

  std::cout <<"Main\n" << std::flush;;
  const char *text="Hello";
#if _WIN64
  HMODULE handle=0;
  int (*fptr)();
#else
  void *handle=0;
  int (*fptr)();
#endif
  int val=-41;

  std::cout <<"Open\n" << std::flush;;

#if _WIN64
  handle=LoadLibrary("plugin_ximea.dll");
#else
  handle=dlopen("./libplugin_test1.so", RTLD_NOW|RTLD_LOCAL);
#endif

  if (handle==0) {
	  printf("Couldn't load: Handle==0");
	  chkerr();
	  return -1;
  };
  
// Error check. What to do if can't load?
  chkerr();
  std::cout << handle << " Ok1\n" << std::flush;;

#if _WIN64
  *(void **)(&fptr)=GetProcAddress(handle,"init");
#else
  *(void **)(&fptr) = dlsym(handle,"do_process");
#endif

  chkerr();
  std::cout <<" Ok2\n" << std::flush;;

#if _WIN64
  val=(*fptr)();
#else
  val=(*fptr)();
#endif

  std::cout <<" Ok3\n" << std::flush;;
  std::cout << val;

#if _WIN64
#else
  dlclose(handle);
#endif

  chkerr();
}

