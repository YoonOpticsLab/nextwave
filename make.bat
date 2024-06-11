rem Run me from the build directory of the base source tree
rem cd build; cmake ..   // need to do this once
cmake --build build

copy build\plugin_ximea\debug\plugin_ximea.dll build\debug
rem copy build\plugin_centroiding\debug\plugin_centroiding.lib build\debug

REM Don't ASK. WTF Windows.
copy C:\nextwave\build\Debug\plugin_rawplayer.dll build\debug
rem copy C:\nextwave\build\Debug\plugin_rawplayer.dll build\debug