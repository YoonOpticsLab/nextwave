//=============================================================================
// Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

/**
*  @example Acquisition.cpp
*
*  @brief Acquisition.cpp shows how to acquire images. It relies on
*  information provided in the Enumeration example. Also, check out the
*  ExceptionHandling and NodeMapInfo examples if you haven't already.
*  ExceptionHandling shows the handling of standard and Spinnaker exceptions
*  while NodeMapInfo explores retrieving information from various node types.
*
*  This example touches on the preparation and cleanup of a camera just before
*  and just after the acquisition of images. Image retrieval and conversion,
*  grabbing image data, and saving images are all covered as well.
*
*  Once comfortable with Acquisition, we suggest checking out
*  AcquisitionMultipleCamera, NodeMapCallback, or SaveToAvi.
*  AcquisitionMultipleCamera demonstrates simultaneously acquiring images from
*  a number of cameras, NodeMapCallback serves as a good introduction to
*  programming with callbacks and events, and SaveToAvi exhibits video creation.
*
*  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
*  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
*  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
*/

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>

#include <Windows.h>

// For NextWave Plugin
#include "nextwave_plugin.cpp"

#include "nextwave_plugin.hpp" // Just DECL

//#pragma pack(push,1)
//#include "memory_layout.h"
//#pragma pack(pop) // restore previous setting

// Add this directory (right-click on project in solution explorer, etc.)
//#include "C:\Users\drcoates\Documents\code\nextwave\boost_1_83_0"
#include "boost/interprocess/windows_shared_memory.hpp"
#include "boost/interprocess/mapped_region.hpp"
using namespace boost::interprocess;

// TODO: Make this a settable parameter?
#define ACQUIRE_TIMEOUT_MS 100

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

#include "../include/spdlog/spdlog.h"

// UI Socket communication
#include <cstring>
#define CAMERA_SOCKET 50007
#include "socket.cpp"

// Use the following enum and global constant to select whether a software or
// hardware trigger is used.
enum triggerType
{
    SOFTWARE,
    HARDWARE
};

const triggerType chosenTrigger = SOFTWARE;

// This function configures the camera to use a trigger. First, trigger mode is
// set to off in order to select the trigger source. Once the trigger source
// has been selected, trigger mode is then enabled, which has the camera
// capture only a single image upon the execution of the chosen trigger.
int ConfigureTrigger(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING TRIGGER ***" << endl << endl;

    cout << "Note that if the application / user software triggers faster than frame time, the trigger may be dropped "
            "/ skipped by the camera."
         << endl
         << "If several frames are needed per trigger, a more reliable alternative for such case, is to use the "
            "multi-frame mode."
         << endl
         << endl;

    if (chosenTrigger == SOFTWARE)
    {
        cout << "Software trigger chosen..." << endl;
    }
    else if (chosenTrigger == HARDWARE)
    {
        cout << "Hardware trigger chosen..." << endl;
    }

    try
    {
        //
        // Ensure trigger mode off
        //
        // *** NOTES ***
        // The trigger must be disabled in order to configure whether the source
        // is software or hardware.
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << "Trigger mode disabled..." << endl;

        //
        // Set TriggerSelector to FrameStart
        //
        // *** NOTES ***
        // For this example, the trigger selector should be set to frame start.
        // This is the default for most cameras.
        //
        CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
        if (!IsReadable(ptrTriggerSelector) ||
            !IsWritable(ptrTriggerSelector))
        {
            cout << "Unable to get or set trigger selector (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerSelectorFrameStart = ptrTriggerSelector->GetEntryByName("FrameStart");
        if (!IsReadable(ptrTriggerSelectorFrameStart))
        {
            cout << "Unable to get trigger selector (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerSelector->SetIntValue(ptrTriggerSelectorFrameStart->GetValue());

        cout << "Trigger selector set to frame start..." << endl;

        //
        // Select trigger source
        //
        // *** NOTES ***
        // The trigger source must be set to hardware or software while trigger
        // mode is off.
        //
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsReadable(ptrTriggerSource) ||
            !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to get or set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        if (chosenTrigger == SOFTWARE)
        {
            // Set trigger mode to software
            CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
            if (!IsReadable(ptrTriggerSourceSoftware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());

            cout << "Trigger source set to software..." << endl;
        }
        else if (chosenTrigger == HARDWARE)
        {
            // Set trigger mode to hardware ('Line0')
            CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line0");
            if (!IsReadable(ptrTriggerSourceHardware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());

            cout << "Trigger source set to hardware..." << endl;
        }

        //
        // Turn trigger mode on
        //
        // *** LATER ***
        // Once the appropriate trigger source has been set, turn trigger mode
        // on in order to retrieve images using the trigger.
        //

        CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
        if (!IsReadable(ptrTriggerModeOn))
        {
            cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

        // NOTE: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on

        cout << "Trigger mode turned back on..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to a normal state by turning off trigger
// mode.
int ResetTrigger(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn trigger mode back off
        //
        // *** NOTES ***
        // Once all images have been captured, turn trigger mode back off to
        // restore the camera to a clean state.
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Non-fatal error..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Non-fatal error..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << "Trigger mode disabled..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}


// Disables or enables heartbeat on GEV cameras so debugging does not incur timeout errors
int ConfigureGVCPHeartbeat(CameraPtr pCam, bool enable)
{
    //
    // Write to boolean node controlling the camera's heartbeat
    //
    // *** NOTES ***
    // This applies only to GEV cameras.
    //
    // GEV cameras have a heartbeat built in, but when debugging applications the
    // camera may time out due to its heartbeat. Disabling the heartbeat prevents
    // this timeout from occurring, enabling us to continue with any necessary 
    // debugging.
    //
    // *** LATER ***
    // Make sure that the heartbeat is reset upon completion of the debugging.  
    // If the application is terminated unexpectedly, the camera may not locked
    // to Spinnaker indefinitely due to the the timeout being disabled.  When that 
    // happens, a camera power cycle will reset the heartbeat to its default setting.

    // Retrieve TL device nodemap
    INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

    // Retrieve GenICam nodemap
    INodeMap& nodeMap = pCam->GetNodeMap();

    CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
    if (!IsReadable(ptrDeviceType))
    {
        return -1;
    }

    if (ptrDeviceType->GetIntValue() != DeviceType_GigEVision)
    {
        return 0;
    }

    if (enable)
    {
        cout << endl << "Resetting heartbeat..." << endl << endl;
    }
    else
    {
        cout << endl << "Disabling heartbeat..." << endl << endl;
    }

    CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
    if (!IsWritable(ptrDeviceHeartbeat))
    {
        cout << "Unable to configure heartbeat. Continuing with execution as this may be non-fatal..."
            << endl
            << endl;
    }
    else
    {
        ptrDeviceHeartbeat->SetValue(enable);

        if (!enable)
        {
            cout << "WARNING: Heartbeat has been disabled for the rest of this example run." << endl;
            cout << "         Heartbeat will be reset upon the completion of this run.  If the " << endl;
            cout << "         example is aborted unexpectedly before the heartbeat is reset, the" << endl;
            cout << "         camera may need to be power cycled to reset the heartbeat." << endl << endl;
        }
        else
        {
            cout << "Heartbeat has been reset." << endl;
        }
    }

    return 0;
}

int ResetGVCPHeartbeat(CameraPtr pCam)
{
    return ConfigureGVCPHeartbeat(pCam, true);
}

int DisableGVCPHeartbeat(CameraPtr pCam)
{
    return ConfigureGVCPHeartbeat(pCam, false);
}

// Global variable:
CameraPtr pCam = nullptr;
CameraList camList;
unsigned int numCameras;
SystemPtr mySystem;
ImageProcessor processor;

uint16_t nCurrRing = 0; //persist across calls

int Acquire1Image(void)
{
    int result = 0;

    try {
      ImagePtr pResultImage = pCam->GetNextImage(ACQUIRE_TIMEOUT_MS);
    } catch (Spinnaker::Exception& e)
    {
      cout << "Acq1 Error: " << e.what() << endl;
      return -1;
    };

    // Ensure image completion
    if (pResultImage->IsIncomplete())
    {
        // Retrieve and print the image status description
        cout << "Image incomplete: " << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
            << "..." << endl
            << endl;
        return -2;
    }
    else
    {
        const size_t width = pResultImage->GetWidth();
        const size_t height = pResultImage->GetHeight();

        nCurrRing += 1;
        if (nCurrRing >= NW_MAX_FRAMES) nCurrRing = 0;

        ImagePtr convertedImage = processor.Convert(pResultImage, PixelFormat_Mono8);

        //struct shmem_header* pShmem = (struct shmem_header*) shmem_region.get_address();

        gpShmemHeader->lock = (uint8_t)1; // Everyone keep out until we are done!

        // Don't need to write these each time:
        gpShmemHeader->header_version = (uint8_t)NW_HEADER_VERSION;
        gpShmemHeader->dimensions[0] = (uint16_t)height;
        gpShmemHeader->dimensions[1] = (uint16_t)width;
        gpShmemHeader->dimensions[2] = (uint16_t)0;
        gpShmemHeader->dimensions[3] = (uint16_t)0;
        gpShmemHeader->datatype_code = (uint8_t)7;
        gpShmemHeader->max_frames = (uint8_t)NW_MAX_FRAMES;

        // For current frame:
        gpShmemHeader->current_frame = (uint8_t)nCurrRing;
        gpShmemHeader->timestamps[nCurrRing] = (uint8_t)NW_STATUS_READ;
        gpShmemHeader->timestamps[nCurrRing] = convertedImage->GetTimeStamp();

        memcpy( ((uint8_t *)(shmem_region2.get_address())+height*width*nCurrRing),
            (void*)convertedImage->GetData(),
            height*width);

        gpShmemHeader->lock = (uint8_t)0; // Keep out until we are done!

//spdlog::info("Acquired: {}",nCurrRing);
    }

    pResultImage->Release();
    return 0;
}

int AcquireImage(CameraPtr pCam) //, INodeMap& nodeMap, INodeMap& nodeMapTLDevice)
{
    int result = 0;

		INodeMap& nodeMap = pCam->GetNodeMap();
    // Execute software trigger
    CCommandPtr ptrSoftwareTriggerCommand = nodeMap.GetNode("TriggerSoftware");
    if (!IsWritable(ptrSoftwareTriggerCommand))
      {
        cout << "Unable to execute trigger. Aborting..." << endl;
        return -1;
      }

  // Fire software trigger to start acquisition of new image
  ptrSoftwareTriggerCommand->Execute();

  if (Acquire1Image()<0) { // Some kind of error, try to restart device
    pCam->EndAcquisition();
    pCam->BeginAcquisition();
    if (Acquire1Image()==0) { 
      spdlog::info("End/begin succeeded.");
    }
  }

  return result;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(INodeMap& nodeMap)
{
    int result = 0;
    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsReadable(category))
        {
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it)
            {
                const CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;
            }
        }
        else
        {
            cout << "Device control information not available." << endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result;

    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure heartbeat for GEV camera
#ifdef _DEBUG
        result = result | DisableGVCPHeartbeat(pCam);
#else
        result = result | ResetGVCPHeartbeat(pCam);
#endif
	}
	catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
	}
	return result;
}
// This function configures a custom exposure time. Automatic exposure is turned
// off in order to allow for the customization, and then the custom setting is
// applied.
int ConfigureExposure(INodeMap& nodeMap, double exposureTimeToSet)
{
    int result = 0;
    cout << endl << endl << "*** CONFIGURING EXPOSURE ***" << endl << endl;
    try
    {
        //
        // Turn off automatic exposure mode
        //
        // *** NOTES ***
        // Automatic exposure prevents the manual configuration of exposure
        // time and needs to be turned off. Some models have auto-exposure
        // turned off by default
        //
        // *** LATER ***
        // Exposure time can be set automatically or manually as needed. This
        // example turns automatic exposure off to set it manually and back
        // on in order to return the camera to its default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (IsReadable(ptrExposureAuto) &&
            IsWritable(ptrExposureAuto))
        {
            CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
            if (IsReadable(ptrExposureAutoOff))
            {
                ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());
                cout << "Automatic exposure disabled..." << endl;
            }
        }
        else 
        {
            CEnumerationPtr ptrAutoBright = nodeMap.GetNode("autoBrightnessMode");
            if (!IsReadable(ptrAutoBright) ||
                !IsWritable(ptrAutoBright))
            {
                cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
                return -1;
            }
            cout << "Unable to disable automatic exposure. Expected for some models... " << endl;
            cout << "Proceeding..." << endl;
            result = 1;
        }
        //
        // Set exposure time manually; exposure time recorded in microseconds
        //
        // *** NOTES ***
        // The node is checked for availability and writability prior to the
        // setting of the node. Further, it is ensured that the desired exposure
        // time does not exceed the maximum. Exposure time is counted in
        // microseconds. This information can be found out either by
        // retrieving the unit with the GetUnit() method or by checking SpinView.
        //
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime) ||
            !IsWritable(ptrExposureTime))
        {
            cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
            return -1;
        }
        // Ensure desired exposure time does not exceed the maximum
        const double exposureTimeMax = ptrExposureTime->GetMax();
        
        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }
		        //CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        ptrExposureTime->SetValue(exposureTimeToSet);
        cout << std::fixed << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
    return result;
}

		
// Example entry point; please see Enumeration example for more in-depth
// comments on preparing and cleaning up the system.
DECL init(void)
{
	 plugin_access_shmem();
	 
    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    mySystem = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = mySystem->GetLibraryVersion();
    cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
        << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
        << endl;

    // Retrieve list of cameras from the system
    camList = mySystem->GetCameras();

    numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    // Finish if there are no cameras
    if (numCameras == 0)
    {
        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        mySystem->ReleaseInstance();

        cout << "Not enough cameras!" << endl;
        //cout << "Done! Press Enter to exit..." << endl;
        //getchar();

        return -1;
    }

	int result=0;
    //
    // Create shared pointer to camera
    //
    // *** NOTES ***
    // The CameraPtr object is a shared pointer, and will generally clean itself
    // up upon exiting its scope. However, if a shared pointer is created in the
    // same scope that a system object is explicitly released (i.e. this scope),
    // the reference to the shared point must be broken manually.
    //
    // *** LATER ***
    // Shared pointers can be terminated manually by assigning them to nullptr.
    // This keeps releasing the system from throwing an exception.
    //
    try
    {
		pCam = camList[0];
		RunSingleCamera( pCam );

		// Retrieve TL device nodemap
		INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

		// Retrieve GenICam nodemap
		INodeMap& nodeMap = pCam->GetNodeMap();

        //
        // Set acquisition mode to continuous
        //
        // *** NOTES ***
        // Because the example acquires and saves 10 images, setting acquisition
        // mode to continuous lets the example finish. If set to single frame
        // or multiframe (at a lower number of images), the example would just
        // hang. This would happen because the example has been written to
        // acquire 10 images while the camera would have been programmed to
        // retrieve less than that.
        //
        // Setting the value of an enumeration node is slightly more complicated
        // than other node types. Two nodes must be retrieved: first, the
        // enumeration node is retrieved from the nodemap; and second, the entry
        // node is retrieved from the enumeration node. The integer value of the
        // entry node is then set as the new value of the enumeration node.
        //
        // Notice that both the enumeration and the entry nodes are checked for
        // availability and readability/writability. Enumeration nodes are
        // generally readable and writable whereas their entry nodes are only
        // ever readable.
        //
        // Retrieve enumeration node from nodemap
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) ||
            !IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to get or set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        //cout << "Acquisition mode set to continuous..." << endl;

		ConfigureExposure(nodeMap, 80000);

    ConfigureTrigger(nodeMap);

        //
        // Begin acquiring images
        //
        // *** NOTES ***
        // What happens when the camera begins acquiring images depends on the
        // acquisition mode. Single frame captures only a single image, multi
        // frame captures a set number of images, and continuous captures a
        // continuous stream of images. Because the example calls for the
        // retrieval of 10 images, continuous mode has been set.
        //
        // *** LATER ***
        // Image acquisition must be ended when no more images are needed.
        //
        pCam->BeginAcquisition();

        //cout << "Acquiring images..." << endl;

        //
        // Retrieve device serial number for filename
        //
        // *** NOTES ***
        // The device serial number is retrieved in order to keep cameras from
        // overwriting one another. Grabbing image IDs could also accomplish
        // this.
        //
        gcstring deviceSerialNumber("");
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        //
        // Set default image processor color processing method
        //
        // *** NOTES ***
        // By default, if no specific color processing algorithm is set, the image
        // processor will default to NEAREST_NEIGHBOR method.
        //
        //processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);
	}
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
	
	return 0;
}

//PLUGIN_API(flir,process,char *params)
DECL process(char *params) {
    int result = 0;

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        //cout << endl << "Running example for camera " << i << "..." << endl;

        result = result | AcquireImages(pCam);
    }

  pCam=camList.GetByIndex(0);
INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

// Retrieve GenICam nodemap
INodeMap& nodeMap = pCam->GetNodeMap();
  
  char *msg=socket_check(CAMERA_SOCKET);
  if (msg!=NULL) {
    //spdlog::info("RAW: {}",msg);
    if (msg[0]=='E'){
        double dVal = atof(msg+2);
        //spdlog::info("exposure: {}",dVal);
		CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        ptrExposureTime->SetValue(dVal);
      }

    if (msg[0]=='G'){
        double dVal = atof(msg+2);
		CFloatPtr ptrExposureTime = nodeMap.GetNode("Gain");
        ptrExposureTime->SetValue(dVal);
      }	  
	};
  
	return 0;
}

DECL plugin_close(void)
{
	int result=0;
	
	pCam->EndAcquisition();
			
    INodeMap& nodeMap = pCam->GetNodeMap();
    ResetTrigger(nodeMap);

	try {
#ifdef _DEBUG
        // Reset heartbeat for GEV camera
        result = result | ResetGVCPHeartbeat(pCam);
#endif

        // Deinitialize camera
        pCam->DeInit();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
	

    //
    // Release reference to the camera
    //
    // *** NOTES ***
    // Had the CameraPtr object been created within the for-loop, it would not
    // be necessary to manually break the reference because the shared pointer
    // would have automatically cleaned itself up upon exiting the loop.
    //
    pCam = nullptr;

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    mySystem->ReleaseInstance();

    //cout << endl << "Done! Press Enter to exit..." << endl;
    //getchar();

    return 0;
}
