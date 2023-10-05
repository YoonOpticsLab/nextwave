#include <iostream>
#include "nextwave_plugin.hpp"

extern "C" void init(void)
{
  std::cout <<"P1 init";
};

extern "C" void do_process(const char* which_buffer)
{
  std::cout <<"P1 do_process";
};

extern "C" void set_params(const char* settings_as_json)
{
  std::cout <<"P1 set_params";
};

extern "C" void get_info(const char* which_info)
{
  std::cout <<"P1 get_info";
};
