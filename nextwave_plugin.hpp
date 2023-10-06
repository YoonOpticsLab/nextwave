#if _WIN64
#include <windows.h>
#define DECL extern "C"  __declspec(dllexport) int
#else
#define DECL extern "C" int
#endif

//namespace NextWave {

  //DECL init(void);
  //DECL do_process(const char* which_buffer);
  //DECL set_params(const char* settings_as_json);
  //DECL get_info(const char* which_info);
//}