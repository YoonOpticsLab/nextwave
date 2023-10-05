namespace NextWave {
 __declspec(dllexport) int init(void);
 __declspec(dllexport) int do_process(void);
 __declspec(dllexport) int set_params(const char* settings_as_json);
 __declspec(dllexport) char* info_as_json get_info(const char* which_as_json);
}

