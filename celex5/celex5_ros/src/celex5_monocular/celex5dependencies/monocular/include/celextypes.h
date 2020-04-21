/*
* Copyright (c) 2017-2018 CelePixel Technology Co. Ltd. All Rights Reserved
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

#ifndef CELEXTYPES_H
#define CELEXTYPES_H

#include <stdint.h>
#include <ctime>

#define SLIDER_DELAY   100
#define MAX_LOG_LINES  100

#define MAX_PAGE_COUNT 100000

#define PIPEOUT_TIMER  10
#define EVENT_SIZE     4
#define PIXELS_PER_COL 768
#define PIXELS_PER_ROW 640
#define PIXELS_NUMBER  491520

#define CELEX5_COL            1280
#define CELEX5_ROW            800
#define  CELEX5_PIXELS_NUMBER 1024000

#define MIRROR_VERTICAL     1
#define MIRROR_HORIZONTAL   1

#define FILE_CELEX5_CFG              "CeleX5_Commands_MIPI.xml"
#define FILE_CELEX5_CFG_MIPI         "cfg_mp"
#define FILE_CELEX5_CFG_MIPI_WRIE    "cfg_mp_wire"

#define SEQUENCE_LAYOUT_WIDTH 3 //7
#define SLIDER_LAYOUT_WIDTH   1 //4
#define DIALOG_LAYOUT_WIDTH   2

#define FPN_CALCULATION_TIMES 5

#define TIMER_CYCLE 25000000  //1s
#define HARD_TIMER_CYCLE 262144  //2^17=131072; 2^18=262144

typedef struct EventData
{
	uint16_t    col;
	uint16_t    row;
	uint16_t    adc; //Event_Off_Pixel_Timestamp_Mode: adc is 0; Event Intensity Mode: adc is "Intensity"; Event_In_Pixel_Timestamp_Mode: adc is "Optical-flow T"
	int16_t     polarity; //-1: intensity weakened; 1: intensity is increased; 0 intensity unchanged
	uint32_t    t_off_pixel; //it will be reset after the end of a frame
	uint64_t    t_off_pixel_increasing; //it won't be reset, it's a monotonically increasing value
	uint32_t    t_in_pixel_ramp_no;
	uint64_t    t_in_pixel_increasing;
} EventData;

typedef enum EventShowType
{
	EventShowByTime = 0,
	EventShowByCount = 1,
	EventShowByRowCycle = 2,
}EventShowType;

typedef enum PlaybackState {
	NoBinPlaying = 0,
	Playing,
	BinReadFinished,
	PlayFinished,
	Replay
}PlaybackState;

typedef struct IMUData {
	double			x_GYROS;
	double			y_GYROS;
	double			z_GYROS;
	uint32_t		t_GYROS;
	double			x_ACC;
	double			y_ACC;
	double			z_ACC;
	uint32_t		t_ACC;
	double			x_MAG;
	double			y_MAG;
	double			z_MAG;
	uint32_t		t_MAG;
	double			x_TEMP;
	uint64_t		frameNo;
	std::time_t     time_stamp;
} IMUData;


typedef struct IMURawData
{
	uint8_t       imu_data[20];
	std::time_t   time_stamp;
} IMURawData;

#endif // CELEXTYPES_H
