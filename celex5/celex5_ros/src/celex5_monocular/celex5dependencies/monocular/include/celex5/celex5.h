/*
* Copyright (c) 2017-2018  CelePixel Technology Co. Ltd.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef CELEX5_H
#define CELEX5_H

#include <stdint.h>
#include <vector>
#include <map>
#include <list>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../celextypes.h"

#ifdef _WIN32
#ifdef CELEX_API_EXPORTS
#define CELEX_EXPORTS __declspec(dllexport)
#else
#define CELEX_EXPORTS __declspec(dllimport)
#endif
#else
#if defined(CELEX_LIBRARY)
#define CELEX_EXPORTS
#else
#define CELEX_EXPORTS
#endif
#endif

using namespace std;

class CeleDriver;
class CeleX5DataProcessor;
class HHSequenceMgr;
class DataProcessThreadEx;
class DataReaderThread;
class CX5SensorDataServer;
class CommandBase;
class DataRecorder;
class CELEX_EXPORTS CeleX5
{
public:
	enum DeviceType {
		Unknown_Devive = 0,
		CeleX5_MIPI = 1,
		CeleX5_OpalKelly = 2,
	};

	enum CeleX5Mode {
		Unknown_Mode = -1,
		Event_Off_Pixel_Timestamp_Mode = 0,	//Using Event_Off_Pixel_Timestamp_Mode, Event_Address_Only_Mode is deprecated.
		Event_In_Pixel_Timestamp_Mode = 1,	//Using Event_In_Pixel_Timestamp_Mode, Event_Optical_Flow_Mode is deprecated.
		Event_Intensity_Mode = 2,	
		Full_Picture_Mode = 3,
		Optical_Flow_Mode = 4,	//Using Optical_Flow_Mode, Full_Optical_Flow_S_Mode is deprecated.
		Optical_Flow_FPN_Mode = 5,	//Using Optical_Flow_FPN_Mode, Full_Optical_Flow_Test_Mode is deprecated.
		Multi_Read_Optical_Flow_Mode = 6,	//Using Multi_Read_Optical_Flow_Mode, Full_Optical_Flow_M_Mode  is deprecated.
	};

	enum emEventPicType {
		EventBinaryPic = 0,
		EventAccumulatedPic = 1,
		EventGrayPic = 2,
		EventCountPic = 3,
		EventDenoisedBinaryPic = 4,
		EventSuperimposedPic = 5,
		EventDenoisedCountPic = 6
	};

	enum emFullPicType {
		Full_Optical_Flow_Pic = 0,
		Full_Optical_Flow_Speed_Pic = 1,
		Full_Optical_Flow_Direction_Pic = 2
	};

	typedef struct CfgInfo
	{
		std::string name;
		uint32_t    min;
		uint32_t    max;
		uint32_t    value;
		uint32_t    step;
		int16_t     high_addr;
		int16_t     middle_addr;
		int16_t     low_addr;
	} CfgInfo;

	typedef struct BinFileAttributes
	{
		uint8_t    data_type; //bit0: 0: fixed mode; 1: loop mode; bit1: 0: no IMU data; 1: has IMU data
		uint8_t    loopA_mode;
		uint8_t    loopB_mode;
		uint8_t    loopC_mode;
		uint8_t    event_data_format;
		uint8_t    hour;
		uint8_t    minute;
		uint8_t    second;
		uint32_t   package_count;
	} BinFileAttributes;

	CeleX5();
	~CeleX5();

	bool openSensor(DeviceType type);
	bool isSensorReady();

	/*
	* Get Sensor raw data interfaces
	* If you don't care about IMU data, you can use the first getMIPIData interface, 
	* otherwise you need to use the second getMIPIData interface.
	*/
	void getMIPIData(vector<uint8_t> &buffer);
	void getMIPIData(vector<uint8_t> &buffer, std::time_t& time_stamp_end, vector<IMURawData>& imu_data);

	/*
	* Parse Sensor raw data interfaces
	* If you don't care about IMU data, you can use the first parseMIPIData interface,
	* otherwise you need to use the second parseMIPIData interface.
	*/
	void parseMIPIData(uint8_t* pData, int dataSize);
	void parseMIPIData(uint8_t* pData, int dataSize, std::time_t time_stamp_end, vector<IMURawData> imu_data);

	/* 
	* Enable/Disable the Create Image Frame module
	* If you just want to obtain (x,y,A,t) array (don't need frame data), you cound disable this function to imporve performance.
	*/
	void disableFrameModule(); 
	void enableFrameModule();
	bool isFrameModuleEnabled();

	/*
	* Enable/Disable the Event Stream module
	* If you just want to event frame (don't need (x,y,A,t) stream), you cound disable this function to imporve performance.
	*/
	void disableEventStreamModule();
	void enableEventStreamModule();
	bool isEventStreamEnabled();

	/*
	* Disable/Enable the IMU module
	* If you don't want to obtain IMU data, you cound disable this function to imporve performance.
	*/
	void disableIMUModule();
	void enableIMUModule();
	bool isIMUModuleEnabled();

	/*
	* Get Full-frame pic buffer or mat
	*/
	void getFullPicBuffer(unsigned char* buffer);
	void getFullPicBuffer(unsigned char* buffer, std::time_t& time_stamp);
	cv::Mat getFullPicMat();

	/*
	* Get event pic buffer or mat
	*/
	void getEventPicBuffer(unsigned char* buffer, emEventPicType type = EventBinaryPic);
	void getEventPicBuffer(unsigned char* buffer, std::time_t& time_stamp, emEventPicType type = EventBinaryPic);
	cv::Mat getEventPicMat(emEventPicType type);

	/*
	* Get optical-flow pic buffer or mat
	*/
	void getOpticalFlowPicBuffer(unsigned char* buffer, emFullPicType type = Full_Optical_Flow_Pic);
	void getOpticalFlowPicBuffer(unsigned char* buffer, std::time_t& time_stamp, emFullPicType type = Full_Optical_Flow_Pic);
	cv::Mat getOpticalFlowPicMat(emFullPicType type);

	/*
	* Get event data vector interfaces
	*/
	bool getEventDataVector(std::vector<EventData> &vector);
	bool getEventDataVector(std::vector<EventData> &vector, uint64_t& frameNo);
	bool getEventDataVectorEx(std::vector<EventData> &vector, std::time_t& time_stamp, bool bDenoised = false);

	/*
	* Get IMU Data
	*/
	int getIMUData(std::vector<IMUData>& data);

	/*
	* Set and get sensor mode (fixed mode)
	*/
	void setSensorFixedMode(CeleX5Mode mode);
	CeleX5Mode getSensorFixedMode();

	/*
	* Set and get sensor mode (Loop mode)
	*/
	void setSensorLoopMode(CeleX5Mode mode, int loopNum); //LopNum = 1/2/3
	CeleX5Mode getSensorLoopMode(int loopNum); //LopNum = 1/2/3
	void setLoopModeEnabled(bool enable);
	bool isLoopModeEnabled();

	/*
	* Set fpn file to be used in Full_Picture_Mode or Event_Intensity_Mode.
	*/
	bool setFpnFile(const std::string& fpnFile);

	/*
	* Generate fpn file
	*/
	void generateFPN(std::string fpnFile);

	/*
	* Clock
	* By default, the CeleX-5 sensor works at 100 MHz and the range of clock rate is from 20 to 100, step is 10.
	*/
	void setClockRate(uint32_t value); //unit: MHz
	uint32_t getClockRate(); //unit: MHz

	/*
	* Threshold
	* The threshold value only works when the CeleX-5 sensor is in the Event Mode.
	* The large the threshold value is, the less pixels that the event will be triggered (or less active pixels).
	* The value could be adjusted from 50 to 511, and the default value is 171.
	*/
	void setThreshold(uint32_t value);
	uint32_t getThreshold();

	/*
	* Brightness
	* Configure register parameter, which controls the brightness of the image CeleX-5 sensor generated.
	* The value could be adjusted from 0 to 1023.
	*/
	void setBrightness(uint32_t value);
	uint32_t getBrightness();

	/*
	* ISO Level
	*/
	void setISOLevel(uint32_t value);
	uint32_t getISOLevel();
	uint32_t getISOLevelCount();

	/*
	* Get the frame time of full-frame picture mode
	*/
	uint32_t getFullPicFrameTime();

	/*
	* Set and get event frame time
	*/
	void setEventFrameTime(uint32_t value); //unit: microsecond
	uint32_t getEventFrameTime();

	/*
	* Set and get frame time of optical-flow mode
	*/
	void setOpticalFlowFrameTime(uint32_t value); //hardware parameter, unit: microsecond
	uint32_t getOpticalFlowFrameTime();

	/* 
	* Loop mode: mode duration
	*/
	void setEventDuration(uint32_t value);
	void setPictureNumber(uint32_t num, CeleX5Mode mode);

	/*
	* Control Sensor interfaces
	*/
	void reset(); //soft reset sensor
	void pauseSensor();
	void restartSensor();
	void stopSensor();

	/*
	* Get the serial number of the sensor, and each sensor has a unique serial number.
	*/
	std::string getSerialNumber();

	/*
	* Get the firmware version of the sensor.
	*/
	std::string getFirmwareVersion();

	/*
	* Get the release date of firmware.
	*/
	std::string getFirmwareDate();
	
	/*
	* Set and get event show method
	*/
	void setEventShowMethod(EventShowType type, int value);
	EventShowType getEventShowMethod();

	/*
	* Set and get rotate type
	*/
	void setRotateType(int type);
	int getRotateType();

	/*
	* Set and get event count stop
	*/
	void setEventCountStepSize(uint32_t size);
	uint32_t getEventCountStepSize();

	/*
	* bit7:0~99, bit6:101~199, bit5:200~299, bit4:300~399, bit3:400~499, bit2:500~599, bit1:600~699, bit0:700~799
	* if rowMask = 240 = b'11110000, 0~399 rows will be closed.
	*/
	void setRowDisabled(uint8_t rowMask);

	/*
	* Whether to display the images when recording
	*/
	void setShowImagesEnabled(bool enable);
	
	/* 
	* Set and get event data format
	*/
	void setEventDataFormat(int format); //0: format 0; 1: format 1; 2: format 2
	int getEventDataFormat();

	void setEventFrameStartPos(uint32_t value); //unit: minisecond

	/*
	* Disable/Enable AntiFlashlight function.
	*/
	void setAntiFlashlightEnabled(bool enabled);

	/*
	* Disable/Enable Auto ISP function.
	*/
	void setAutoISPEnabled(bool enable);
	bool isAutoISPEnabled();
	void setISPThreshold(uint32_t value, int num);
	void setISPBrightness(uint32_t value, int num);

	/*
	* Start/Stop recording raw data.
	*/
	void startRecording(std::string filePath);
	void stopRecording();

	/*
	* Playback Interfaces
	*/
	bool openBinFile(std::string filePath);
	bool readBinFileData();
	uint32_t getTotalPackageCount();
	uint32_t getCurrentPackageNo();
	void setCurrentPackageNo(uint32_t value);
	BinFileAttributes getBinFileAttributes();
	void replay();
	void play();
	void pause();
	PlaybackState getPlaybackState();
	void setPlaybackState(PlaybackState state);
	void setIsPlayBack(bool state);

	CX5SensorDataServer* getSensorDataServer();
	
	DeviceType getDeviceType();

	uint32_t getFullFrameFPS();

	/*
	* Obtain the number of events that being produced per second.
	* Unit: events per second
	*/
	uint32_t getEventRate(); 

	/*
	* Sensor Configures
	*/
	map<string, vector<CfgInfo> > getCeleX5Cfg();
	map<string, vector<CfgInfo> > getCeleX5CfgModified();
	void writeRegister(int16_t addressH, int16_t addressM, int16_t addressL, uint32_t value);
	CfgInfo getCfgInfoByName(string csrType, string name, bool bDefault);
	void writeCSRDefaults(string csrType);
	void modifyCSRParameter(string csrType, string cmdName, uint32_t value);

	//--- for test ---
	int  denoisingMaskByEventTime(const cv::Mat& countEventImg, double timelength, cv::Mat& denoiseMaskImg);
	void saveFullPicRawData();

	void calDirectionAndSpeedEx(cv::Mat pBuffer, cv::Mat &speedBuffer, cv::Mat &dirBuffer);

private:
	bool configureSettings(DeviceType type);
	//for write register
	void wireIn(uint32_t address, uint32_t value, uint32_t mask);
	void writeRegister(CfgInfo cfgInfo);
	void setALSEnabled(bool enable);
	bool isALSEnabled();
	//
	void enterCFGMode();
	void enterStartMode();
	void disableMIPI();
	void enableMIPI();
	void clearData();

private:
	CeleDriver*                    m_pCeleDriver;
	CeleX5DataProcessor*           m_pDataProcessor;
	HHSequenceMgr*                 m_pSequenceMgr;
	DataProcessThreadEx*           m_pDataProcessThread;
	DataRecorder*                  m_pDataRecorder;
	//
	map<string, vector<CfgInfo> >  m_mapCfgDefaults;
	map<string, vector<CfgInfo> >  m_mapCfgModified;
	//
	unsigned char*                 m_pReadBuffer;
	uint8_t*                       m_pDataToRead;

	std::ifstream                  m_ifstreamPlayback;	//playback

	bool                           m_bLoopModeEnabled;
	bool                           m_bALSEnabled;

	uint32_t                       m_uiBrightness;
	uint32_t                       m_uiThreshold;
	uint32_t                       m_uiClockRate;
	uint32_t                       m_uiLastClockRate; // for test
	int                            m_iEventDataFormat;
	uint32_t                       m_uiPackageCount;
	uint32_t                       m_uiTotalPackageCount;
	vector<uint64_t>               m_vecPackagePos;
	bool                           m_bFirstReadFinished;
	BinFileAttributes              m_stBinFileHeader;
	DeviceType                     m_emDeviceType;
	uint32_t                       m_uiPackageCounter;
	uint32_t                       m_uiPackageCountPS;
	uint32_t                       m_uiPackageTDiff;
	uint32_t                       m_uiPackageBeginT;
	bool                           m_bAutoISPEnabled;
	uint32_t                       m_arrayISPThreshold[3];
	uint32_t                       m_arrayBrightness[4];
	uint32_t                       m_uiAutoISPRefreshTime;
	uint32_t                       m_uiISOLevel; //range: 1 ~ 6

	uint32_t                       m_uiOpticalFlowFrameTime;

	int							   m_iRotateType;	//rotate param
	bool                           m_bClockAutoChanged;
	uint32_t                       m_uiISOLevelCount; //4 or 6

	bool                           m_bSensorReady;
	bool                           m_bShowImagesEnabled;
	bool                           m_bAutoISPFrofileLoaded;
};

#endif // CELEX5_H