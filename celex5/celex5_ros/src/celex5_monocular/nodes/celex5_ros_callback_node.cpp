#include "celex5_ros.h"

static const std::string OPENCV_WINDOW = "Image window";
namespace celex_ros_callback {
class CelexRosCallBackNode : public CeleX5DataManager {
public:
  // private ROS node handle
  ros::NodeHandle node_;

  ros::Publisher data_pub_;
  ros::Publisher image_pub_;

  ros::Subscriber data_sub_;

  // custom celex5 message type
  celex5_msgs::eventVector event_vector_;

  // parameters
  std::string celex_mode_, event_pic_type_;
  int threshold_, clock_rate_;

  CX5SensorDataServer *m_pServer_;
  CeleX5 *celex_;
  celex_ros::CelexRos celexRos_;

  CelexRosCallBackNode(CX5SensorDataServer *pServer) : node_("~") {
    m_pServer_ = pServer;
    m_pServer_->registerData(this, CeleX5DataManager::CeleX_Frame_Data);

    image_pub_ = node_.advertise<sensor_msgs::Image>("/imgshow", 10);

    // advertise custom celex5 data and subscribe the data
    data_pub_ = node_.advertise<celex5_msgs::eventVector>("celex5", 10);
    data_sub_ = node_.subscribe("celex5", 1000,
                                &CelexRosCallBackNode::celexDataCallback, this);

    // grab the parameters
    node_.param<std::string>("celex_mode", celex_mode_,
                             "Event_Off_Pixel_Timestamp_Mode");
    node_.param<std::string>("event_pic_type", event_pic_type_,
                             "EventBinaryPic");

    node_.param<int>("threshold", threshold_, 170);   // 0-1024
    node_.param<int>("clock_rate", clock_rate_, 100); // 0-100

    // create the display windows
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~CelexRosCallBackNode() {
    m_pServer_->unregisterData(this, CeleX5DataManager::CeleX_Frame_Data);
    delete celex_;
    cv::destroyWindow(OPENCV_WINDOW);
  }

  // overrides the update operation
  virtual void onFrameDataUpdated(CeleX5ProcessedData *pSensorData);

  // subscribe callback function
  void celexDataCallback(const celex5_msgs::eventVector &msg);

  void setCeleX5(CeleX5 *pcelex);
  bool spin();
};

void CelexRosCallBackNode::celexDataCallback(
    const celex5_msgs::eventVector &msg) {
  ROS_INFO("I heard celex5 data size: [%d]",msg.vectorLength);

  // display the image
  if (msg.vectorLength > 0) {
    cv::Mat mat = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
    for (int i = 0; i < msg.vectorLength; i++) {
      mat.at<uchar>(MAT_ROWS - msg.events[i].x - 1,
                    MAT_COLS - msg.events[i].y - 1) = msg.events[i].brightness;
    }
    cv::imshow(OPENCV_WINDOW, mat);
    cv::waitKey(60);
  }
}

void CelexRosCallBackNode::setCeleX5(CeleX5 *pcelex) {
  celex_ = pcelex;

  celex_->setThreshold(threshold_);
  CeleX5::CeleX5Mode mode;
  if (celex_mode_ == "Event_Off_Pixel_Timestamp_Mode")
    mode = CeleX5::Event_Off_Pixel_Timestamp_Mode;
  celex_->setSensorFixedMode(mode);
}

void CelexRosCallBackNode::onFrameDataUpdated(
    CeleX5ProcessedData *pSensorData) {
  celexRos_.grabEventData(celex_, event_vector_, 0.10);
  data_pub_.publish(event_vector_);
  event_vector_.events.clear();

  // get sensor image and publish it, you can use the RVIZ to subscribe the topic "/imgshow"
  cv::Mat image =
      cv::Mat(800, 1280, CV_8UC1,
              pSensorData->getEventPicBuffer(CeleX5::EventBinaryPic));
  sensor_msgs::ImagePtr msg =
      cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
  image_pub_.publish(msg);
}

bool CelexRosCallBackNode::spin() {
  ros::Rate loop_rate(60);
  while (node_.ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return true;
}
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "celex_monocular_callback");

  CeleX5 *pCelex_;
  pCelex_ = new CeleX5;

  if (NULL == pCelex_)
    return 0;
  pCelex_->openSensor(CeleX5::CeleX5_MIPI);

  celex_ros_callback::CelexRosCallBackNode *cr =
      new celex_ros_callback::CelexRosCallBackNode(
          pCelex_->getSensorDataServer());

  cr->setCeleX5(pCelex_);

  cr->spin();

  return EXIT_SUCCESS;
}
