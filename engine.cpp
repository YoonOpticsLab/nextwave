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
#else
// On unix, for dl* functions:
#include <dlfcn.h>
#endif
// Hints: https://stackoverflow.com/questions/11741580/dlopen-loadlibrary-on-same-application

void chkerr(void)
{
#if _WIN64
#else
  char *errstr = dlerror();
  if (errstr != NULL) {
    std::cout << "Error: " << errstr << "\n" << std::flush;
  }
#endif
}

int main()
{
  std::cout <<"Main\n" << std::flush;;
  const char *text="Hello";
#if _WIN64
  HMODULE handle=0;
  int (*fptr)(const char*);
#else
  void *handle=0;
  int (*fptr)(const char *);
#endif
  int val=-41;

  std::cout <<"Open\n" << std::flush;;

#if _WIN64
  handle=LoadLibrary("plugin_test1.dll");
#else
  handle=dlopen("./libplugin_test1.so", RTLD_NOW|RTLD_LOCAL);
#endif

// Error check. What to do if can't load?
  chkerr();
  std::cout << handle << " Ok1\n" << std::flush;;

#if _WIN64
  *(void **)(&fptr)=GetProcAddress(handle,"set_params");
#else
  *(void **)(&fptr) = dlsym(handle,"do_process");
#endif

  chkerr();
  std::cout <<" Ok2\n" << std::flush;;

#if _WIN64
  val=(*fptr)(text);
#else
  val=(*fptr)(text);
#endif

  std::cout <<" Ok3\n" << std::flush;;
  std::cout << val;

#if _WIN64
#else
  dlclose(handle);
#endif

  chkerr();
}

