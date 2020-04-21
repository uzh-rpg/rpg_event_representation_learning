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

#ifndef CELEX5_DATA_MANAGER_H
#define CELEX5_DATA_MANAGER_H

#include "celex5processeddata.h"

class CeleX5DataManager
{
public:
	enum emDataType {
		Default = 0,
		CeleX_Frame_Data = 1
	};

	CeleX5DataManager() { }
	~CeleX5DataManager() { }

	virtual void onFrameDataUpdated(CeleX5ProcessedData* data) = 0;
};

class CX5Subject
{
public:
	virtual void registerData(CeleX5DataManager* observer, CeleX5DataManager::emDataType type) = 0;
	virtual void unregisterData(CeleX5DataManager* observer, CeleX5DataManager::emDataType type) = 0;
	virtual void notify(CeleX5DataManager::emDataType dataType) = 0;
};

class CX5SensorDataServer : public CX5Subject
{
public:
	CX5SensorDataServer() : m_pObserver(NULL), m_pCX5ProcessedData(NULL)
	{
	}
	virtual ~CX5SensorDataServer()
	{
	}

	void registerData(CeleX5DataManager* observer, CeleX5DataManager::emDataType type)
	{
		m_pObserver = observer;
		m_listDataType.push_back(type);
	}

	void unregisterData(CeleX5DataManager* observer, CeleX5DataManager::emDataType type)
	{
		if (observer == m_pObserver)
		{
			m_listDataType.remove(type);
		}
	}

	void notify(CeleX5DataManager::emDataType dataType)
	{
		if (m_pObserver)
		{
			if (CeleX5DataManager::CeleX_Frame_Data == dataType)
				m_pObserver->onFrameDataUpdated(m_pCX5ProcessedData);
		}
	}
	inline void setCX5SensorData(CeleX5ProcessedData* data) { m_pCX5ProcessedData = data; }
	inline CeleX5ProcessedData* getCX4SensorData() { return m_pCX5ProcessedData; }

private:
	std::list<CeleX5DataManager::emDataType> m_listDataType;
	CeleX5DataManager*      m_pObserver;
	CeleX5ProcessedData*    m_pCX5ProcessedData;
};

#endif // CELEX5_DATA_MANAGER_H
