#include <ctime>
#include "celex5_ros.h"

//static const std::string OPENCV_WINDOW = "Image window";

namespace celex_ros {
class CelexRosNode {
public:
  // private ROS node handle
  ros::NodeHandle node_;

  // custom celex5 message type
  celex5_msgs::eventVector event_vector_;
  celex5_msgs::eventData event_data_;

  // ros::Publisher data_pub_, image_pub_;
  // ros::Subscriber data_sub_;
  ros::Publisher data_pub_;

  // parameters
  std::string celex_mode_;
  int threshold_, clock_rate_, actionMode_;
  std::string bin_path;

  CelexRos celexRos_;
  CeleX5 *celex_;

  CelexRosNode() : node_("~") {

    // image_pub_ = node_.advertise<sensor_msgs::Image>("/celex_image",1);
    data_pub_ = node_.advertise<celex5_msgs::eventData>("celex5_event", 1);
    // data_sub_ = node_.subscribe("celex5_event", 1, &CelexRosNode::celexDataCallback, this);

    // grab the parameters
    node_.param<std::string>("celex_mode", celex_mode_,
                             "Event_Off_Pixel_Timestamp_Mode");

    node_.param<int>("threshold", threshold_, 171);   // 0-1024
    node_.param<int>("clock_rate", clock_rate_, 100); // 0-100
    node_.param<int>("actionMode", actionMode_, 0); // 0-2
    node_.param<std::string>("bin_output_path", bin_path, "NULL");

    // cv::namedWindow(OPENCV_WINDOW);
  }

  // ~CelexRosNode() { cv::destroyWindow(OPENCV_WINDOW); }
  ~CelexRosNode() {}
  // subscribe callback function
  // void celexDataCallback(const celex5_msgs::eventVector &msg);

  bool grabAndSendData();
  void setCeleX5(CeleX5 *pcelex);
  bool spin();
  bool readFromBin();
  bool genBin();
};

// void CelexRosNode::celexDataCallback(const celex5_msgs::eventVector &msg) {
//   // ROS_INFO("I heard celex5 data size: [%d]", msg.vectorLength);
//   if (msg.vectorLength > 0) {
//     cv::Mat mat = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
//     for (int i = 0; i < msg.vectorLength; i++) {
//       mat.at<uchar>(MAT_ROWS - msg.events[i].y - 1,
//                     MAT_COLS - msg.events[i].x - 1) = msg.events[i].brightness;
//     }
//     cv::imshow(OPENCV_WINDOW, mat);
//     cv::waitKey(1);
//   }
// }

bool CelexRosNode::grabAndSendData() {
  celexRos_.grabEventData(celex_, event_vector_, 0.10);
  if(event_vector_.vectorIndex != 0) {
    for (int i=0 ; i<event_vector_.vectorLength ; i++) {
      event_data_.x.push_back(event_vector_.events[i].x);
      event_data_.y.push_back(event_vector_.events[i].y);
      event_data_.timestamp.push_back(event_vector_.events[i].timestamp.toSec());
    }
    data_pub_.publish(event_data_);
    event_data_.x.clear();
    event_data_.y.clear();
    event_data_.timestamp.clear();
  }
  event_vector_.events.clear();

  // get sensor image and publish it
  // cv::Mat image = celex_->getEventPicMat(CeleX5::EventBinaryPic);
  // sensor_msgs::ImagePtr msg =
  //     cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
  // image_pub_.publish(msg);
}

void CelexRosNode::setCeleX5(CeleX5 *pcelex) {
  celex_ = pcelex;

  celex_->setThreshold(threshold_);
  celexRos_.set_frame_interval(celex_->getEventFrameTime()); 
  ROS_INFO("NODE celex_monocular frame time: %f ms", celexRos_.get_frame_interval());

  CeleX5::CeleX5Mode mode;
  if (celex_mode_ == "Event_Off_Pixel_Timestamp_Mode")
    mode = CeleX5::Event_Off_Pixel_Timestamp_Mode;
  else if (celex_mode_ == "Full_Picture_Mode")
    mode = CeleX5::Full_Picture_Mode;
  else if (celex_mode_ == "Event_Intensity_Mode")
    mode = CeleX5::Event_Intensity_Mode;

  celex_->setSensorFixedMode(mode);
  ROS_INFO("NODE celex_monocular sensor mode: %d",  celex_->getSensorFixedMode());
}

bool CelexRosNode::spin() {
  ros::Rate loop_rate(20);

  while (node_.ok()) {
    grabAndSendData();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return true;
}

bool CelexRosNode::readFromBin() {
  ros::Rate loop_rate(300);

  ROS_INFO("bin PATH: %s", bin_path.c_str());
  bool success = celex_->openBinFile(bin_path);
  CeleX5::CeleX5Mode mode = (CeleX5::CeleX5Mode)celex_->getBinFileAttributes().loopA_mode;
  celex_->setSensorFixedMode(mode);
  
  while (node_.ok()) {
    celex_->readBinFileData();
    grabAndSendData();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return true;
}

bool CelexRosNode::genBin() {
  ros::Rate loop_rate(30);
  ROS_INFO("bin PATH: %s", bin_path.c_str());
  int cnt = 0;
  celex_->startRecording(bin_path);
  while (node_.ok() && cnt<600) {
    celexRos_.grabEventData(celex_, event_vector_, 0.10);
    event_vector_.events.clear();
    cnt ++;
    ros::spinOnce();
    loop_rate.sleep();
  }
  celex_->stopRecording();
  return true;
}

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "celex_monocular");

  CeleX5 *pCelex_;
  pCelex_ = new CeleX5;
  if (NULL == pCelex_)
    return 0;
  pCelex_->openSensor(CeleX5::CeleX5_MIPI);

  celex_ros::CelexRosNode cr;
  cr.setCeleX5(pCelex_);
  switch(cr.actionMode_) {
    case 0:
      cr.spin();
      break;
    case 1:
      cr.readFromBin();
      break;
    case 2:
      cr.genBin();
      break;
  }
  return EXIT_SUCCESS;
}
