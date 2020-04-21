#include <ros/ros.h>
#include <ros/package.h>
#include <celex5/celex5.h>
#include <celex5/celex5datamanager.h>
#include <celextypes.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <celex5_msgs/event.h>
#include <celex5_msgs/eventData.h>
#include <celex5_msgs/eventVector.h>
#include <cv_bridge/cv_bridge.h>

#define MAT_ROWS 800
#define MAT_COLS 1280
namespace celex_ros
{
    class CelexRos{
        public:

        CelexRos();
        ~CelexRos();

        int initDevice();
        void setSensorParams();
        void set_frame_interval(uint32_t i);
        double get_frame_interval();
        void grabEventData(CeleX5 *celex, celex5_msgs::eventVector& msg, double max_time_diff);

        private:
        celex5_msgs::event event_;
        ros::Time init_time;
        ros::Duration frame_interval;
        uint32_t frame_cnt;
        uint32_t previous_frame_len, previous_x, previous_y;
    };
}