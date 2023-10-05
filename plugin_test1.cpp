#include <iostream>
#include "nextwave_plugin.hpp"

#if _WIN64
#include <windows.h>
#define DECL extern "C"  __declspec(dllexport) int
#else
#define DECL extern "C" int
#endif
	
DECL init(void)
{
  std::cout <<"P1 init";
  return 42;
};

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
