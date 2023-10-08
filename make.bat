rem Run me from the build directory of the base source tree
cmake --build ..
cmake --build ..\plugin_ximea
copy ..\plugin_ximea\debug\plugin_ximea.dll debug