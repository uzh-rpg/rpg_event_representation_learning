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

#ifndef CELEX5_PROCESSED_DATA_H
#define CELEX5_PROCESSED_DATA_H

#include "celex5.h"

class CeleX5ProcessedData
{
public:
	CeleX5ProcessedData();
	~CeleX5ProcessedData();

	inline unsigned char* getFullPicBuffer() { return m_pFullPic; }
	inline unsigned char* getOpticalFlowPicBuffer(CeleX5::emFullPicType type)
	{
		switch (type)
		{
		case CeleX5::Full_Optical_Flow_Pic:
			return m_pOpticalFlowPic;
		case CeleX5::Full_Optical_Flow_Speed_Pic:
			return m_pOpticalFlowSpeedPic;
		case CeleX5::Full_Optical_Flow_Direction_Pic:
			return m_pOpticalFlowDirectionPic;
		default:
			break;
		}
		return NULL;
	}
	inline void setEventDataVector(std::vector<EventData> eventData)
	{
		m_vectorEventData.clear();
		m_vectorEventData = eventData;
	}
	inline std::vector<EventData> getEventDataVector() { return m_vectorEventData; }
	inline unsigned char* getEventPicBuffer(CeleX5::emEventPicType type) 
	{ 
		switch (type)
		{
		case CeleX5::EventBinaryPic:
			return m_pEventBinaryPic;
		case CeleX5::EventAccumulatedPic:
			return m_pEventAccumulatedPic;
		case CeleX5::EventGrayPic:
			return m_pEventGrayPic;
		case CeleX5::EventCountPic:
			return m_pEventCountPic;
		case CeleX5::EventDenoisedBinaryPic:
			return m_pEventDenoisedBinaryPic;
		case CeleX5::EventSuperimposedPic:
			return m_pEventSuperimposedPic;
		case CeleX5::EventDenoisedCountPic:
			return m_pEventDenoisedCountPic;
		default:
			break;
		}
		return NULL;
	}
	inline CeleX5::CeleX5Mode getSensorMode() { return m_emSensorMode; }
	inline void setSensorMode(CeleX5::CeleX5Mode mode) { m_emSensorMode = mode; }
	inline int getLoopNum() { return m_iLoopNum; }
	inline void setLoopNum(int loopNum) { m_iLoopNum = loopNum; }
	inline void setTemperature(uint16_t temperature) { m_uiTemperature = temperature; }
	inline uint16_t getTemperature() { return m_uiTemperature; }
	inline void setFullFrameFPS(uint16_t fps) { m_uiFullFrameFPS = fps; }
	inline uint16_t getFullFrameFPS() { return m_uiFullFrameFPS; }

private:
	unsigned char*        m_pFullPic;
	unsigned char*        m_pOpticalFlowPic;
	unsigned char*        m_pOpticalFlowSpeedPic;
	unsigned char*        m_pOpticalFlowDirectionPic;
	unsigned char*        m_pEventBinaryPic;
	unsigned char*        m_pEventAccumulatedPic;
	unsigned char*        m_pEventGrayPic;
	unsigned char*        m_pEventCountPic;
	unsigned char*        m_pEventDenoisedBinaryPic;
	unsigned char*		  m_pEventSuperimposedPic;
	unsigned char*        m_pEventDenoisedCountPic;
	CeleX5::CeleX5Mode    m_emSensorMode;
	int                   m_iLoopNum;
	uint16_t              m_uiTemperature;
	uint16_t              m_uiFullFrameFPS;
	std::vector<EventData> m_vectorEventData;       
};

#endif // CELEX5_PROCESSED_DATA_H
