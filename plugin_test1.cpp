#include <iostream>
#include "nextwave_plugin.hpp"
	
DECL init(char *param)
{
  std::cout <<"P1 init" <<"\n";
  return 42;
};

DECL process(char *params)
{
//  std::cout <<"P1 do_process\n" ;
  return 0;
};

DECL close(char *params)
{
  std::cout <<"P1 close\n" ;
  return 0;
};

DECL set_params(char* settings_as_json)
{
  std::cout <<"P1 set_params " << settings_as_json << "\n";
  return 0;
};
 
DECL get_info(char* which_info)
{
  std::cout <<"P1 get_info";
  return 0;
};
