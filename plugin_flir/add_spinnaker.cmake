include_directories(${XIMEA_DIR}/API/xiAPI)

if(UNIX AND NOT APPLE)
    # Linux
    link_libraries(m3api)
elseif(APPLE)
    # macOS
    link_libraries("-F /Library/Frameworks -framework m3api")
    add_compile_options(-F /Library/Frameworks)
else()
    # Windows
    if (${CMAKE_VS_PLATFORM_NAME} STREQUAL "x64")
        link_libraries(Spinnaker_v140)
    else()
        link_libraries(Spinnaker_v140)
    endif()
    add_definitions(-DWIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    link_directories(${SPINNAKER_DIR}/lib64/vs2015)
endif()