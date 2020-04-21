#include <ctime>
#include "celex5_ros.h"

namespace celex_ros {

CelexRos::CelexRos() {}

int CelexRos::initDevice() {
  init_time = ros::Time::now();
  frame_interval = ros::Duration(30/1000000);
  frame_cnt = 0;
  previous_frame_len = 0;
  previous_x = 0;
  previous_y = 0;
}

void CelexRos::set_frame_interval(uint32_t i) {
  frame_interval = ros::Duration(double(i)/1000000);
}

double CelexRos::get_frame_interval() {
  return frame_interval.toSec();
}

void CelexRos::grabEventData(
    CeleX5 *celex,
    celex5_msgs::eventVector &msg,
    double max_time_diff) {
  if (celex->getSensorFixedMode() == CeleX5::Event_Off_Pixel_Timestamp_Mode) {
    std::clock_t timeBegin;
    double timeDiff = 0.0;
    timeBegin = std::clock();

    msg.vectorIndex = 0;
    msg.height = MAT_ROWS;
    msg.width = MAT_COLS;
    msg.vectorLength = 0;
    
    while (timeDiff < max_time_diff) {
      std::vector<EventData> vecEvent;
      celex->getEventDataVector(vecEvent);

      timeDiff = (std::clock() - timeBegin)/(double) CLOCKS_PER_SEC;
      
      uint32_t dataSize = vecEvent.size();
      if (dataSize == 0) {
        continue;
      }
      uint32_t first_x = vecEvent[0].col;
      uint32_t first_y = vecEvent[0].row;
      if (dataSize==previous_frame_len && first_x==previous_x && first_y==previous_y) {
        continue;
      }
      previous_frame_len = dataSize;
      previous_x = first_x;
      previous_y = first_y;

      msg.vectorLength += dataSize;

      cv::Mat mat = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
      for (int i = 0; i < dataSize; i++) {
        mat.at<uchar>(MAT_ROWS - vecEvent[i].row - 1,
                      MAT_COLS - vecEvent[i].col - 1) = 255;
        event_.x = vecEvent[i].col;
        event_.y = vecEvent[i].row;
        event_.timestamp = init_time + ros::Duration(double(vecEvent[i].t_off_pixel)*0.000014);
        msg.events.push_back(event_);
      }
      init_time += frame_interval;
    }
    if (msg.vectorLength == 0)
      return;
    frame_cnt += 1;
    msg.vectorIndex = frame_cnt;
  } else if(celex->getSensorFixedMode() == CeleX5::Event_In_Pixel_Timestamp_Mode) {
    std::vector<EventData> vecEvent;

    celex->getEventDataVector(vecEvent);
    frame_cnt += 1;

    int dataSize = vecEvent.size();
    msg.vectorIndex = frame_cnt;
    msg.height = MAT_ROWS;
    msg.width = MAT_COLS;
    msg.vectorLength = dataSize;

    cv::Mat mat = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
    for (int i = 0; i < dataSize; i++) {
      mat.at<uchar>(MAT_ROWS - vecEvent[i].row - 1,
                    MAT_COLS - vecEvent[i].col - 1) = 255;
      event_.x = vecEvent[i].col;
      event_.y = vecEvent[i].row;
      event_.polarity = vecEvent[i].polarity;
      event_.timestamp = init_time + ros::Duration(double(vecEvent[i].t_off_pixel)*0.000014);
      event_.brightness = 255;
      msg.events.push_back(event_);
    }
    init_time += frame_interval;
  } else if(celex->getSensorFixedMode() == CeleX5::Event_Intensity_Mode) {
    std::vector<EventData> vecEvent;

    celex->getEventDataVector(vecEvent);
    frame_cnt += 1;

    int dataSize = vecEvent.size();
    msg.vectorIndex = frame_cnt;
    msg.height = MAT_ROWS;
    msg.width = MAT_COLS;
    msg.vectorLength = dataSize;

    cv::Mat mat = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
    for (int i = 0; i < dataSize; i++) {
      mat.at<uchar>(MAT_ROWS - vecEvent[i].row - 1,
                    MAT_COLS - vecEvent[i].col - 1) = 255;
      event_.x = vecEvent[i].col;
      event_.y = vecEvent[i].row;
      event_.polarity = vecEvent[i].polarity;
      event_.timestamp = init_time + ros::Duration(double(vecEvent[i].t_off_pixel)*0.000014);
      event_.brightness = 255;
      msg.events.push_back(event_);
    }
    init_time += frame_interval;
    //std::cout.precision(12);
    // std::cout << "Current time: " << std::fixed << init_time.toSec() << std::endl;
  } else {
    msg.vectorLength = 0;
    std::cout << "This mode has no event data. " << std::endl;
  }
}

CelexRos::~CelexRos() {}
}
