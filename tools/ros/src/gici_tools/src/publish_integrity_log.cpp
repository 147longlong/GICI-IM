#include <ros/ros.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <cmath>

struct IntegrityRecord {
    double tx, ty, tz;
    double qx, qy, qz, qw;
    double xpl, ypl, vpl;
};

class IntegrityReplayer {
public:
    IntegrityReplayer(ros::NodeHandle& nh, const std::string& file_path, const std::string& topic_name) {
        pub_pl_ = nh.advertise<geometry_msgs::Vector3Stamped>(topic_name, 100);
        pub_marker_ = nh.advertise<visualization_msgs::Marker>(topic_name + "_marker", 100);
        loadData(file_path);
    }

    void loadData(const std::string& file_path) {
        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            ROS_ERROR("Failed to open file: %s", file_path.c_str());
            return;
        }
        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw, xpl, ypl, vpl;
            ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw >> xpl >> ypl >> vpl;
            if (!ss.fail()) {
                data_map_[timestamp-18] = {tx, ty, tz, qx, qy, qz, qw, xpl, ypl, vpl};
            }
        }
        ROS_INFO("Loaded %lu integrity records.", data_map_.size());
    }

    void publishForTimestamp(ros::Time stamp, const std::string& frame_id, const std::string& child_frame_id = "") {
        if (data_map_.empty()) return;
        
        double query_time = stamp.toSec();
        
        // Find closest timestamp
        auto it = data_map_.lower_bound(query_time);
        if (it == data_map_.end()) it--;
        
        // Check current and previous to find strictly closest
        auto it_prev = it;
        if (it_prev != data_map_.begin()) it_prev--;
        
        double diff_curr = std::abs(it->first - query_time);
        double diff_prev = std::abs(it_prev->first - query_time);
        
        auto best_it = (diff_curr < diff_prev) ? it : it_prev;
        
        // Threshold for association (e.g. 0.1s)
        if (std::abs(best_it->first - query_time) < 0.1) {
            // Publish Vector3Stamped
            geometry_msgs::Vector3Stamped msg;
            msg.header.stamp = stamp; // Use the trigger timestamp for perfect sync
            msg.header.frame_id = frame_id;
            msg.vector.x = best_it->second.ypl;
            msg.vector.y = best_it->second.xpl;
            msg.vector.z = best_it->second.vpl;
            pub_pl_.publish(msg);

            // Publish Marker
            visualization_msgs::Marker marker;
            marker.header.stamp = stamp;
            marker.ns = "protection_level";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            
            if (!child_frame_id.empty()) {
                // Attach to vehicle frame (e.g. base_link)
                marker.header.frame_id = child_frame_id;
                marker.pose.position.x = 0.0;
                marker.pose.position.y = 0.0;
                marker.pose.position.z = 0.0;
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = 0.0;
                marker.pose.orientation.w = 1.0;
            } else {
                // Use global coordinates from file
                marker.header.frame_id = frame_id;
                marker.pose.position.x = best_it->second.tx;
                marker.pose.position.y = best_it->second.ty;
                marker.pose.position.z = best_it->second.tz;
                marker.pose.orientation.x = best_it->second.qx;
                marker.pose.orientation.y = best_it->second.qy;
                marker.pose.orientation.z = best_it->second.qz;
                marker.pose.orientation.w = best_it->second.qw;
            }
            
            // Calculate HPL for color and scale
            double hpl = std::sqrt(best_it->second.xpl * best_it->second.xpl + best_it->second.ypl * best_it->second.ypl);
            
            double r = 0.0, g = 1.0, b = 0.0;
            double scale_factor = 5.0;

            if (hpl < 0.05) {
                r = 0.0; g = 0.392; b = 0.0; // Dark Green #006400
            } else if (hpl < 0.1) {
                r = 0.196; g = 0.804; b = 0.196; // Lime Green #32CD32
                scale_factor = 20.0;
            } else if (hpl < 0.5) {
                r = 0.8; g = 0.8; b = 0.0; // Dark Yellow #CCCC00
                scale_factor = 10.0;
            } else if (hpl < 1.0) {
                r = 1.0; g = 0.647; b = 0.0; // Orange #FFA500
            } else if (hpl < 5.0) {
                r = 1.0; g = 0.271; b = 0.0; // Orange Red #FF4500
                scale_factor = 1.0;
            } else if (hpl < 10.0) {
                r = 1.0; g = 0.0; b = 0.0; // Red #FF0000
                scale_factor = 0.8;
            } else {
                r = 0.502; g = 0.0; b = 0.0; // Maroon #800000
                scale_factor = 0.5; // Do not scale if HPL >= 10.0m
            }

            // Scale: xpl * 2 (diameter) * scale_factor
            marker.scale.x = best_it->second.ypl * 2.0 * scale_factor;
            marker.scale.y = best_it->second.xpl * 2.0 * scale_factor;
            marker.scale.z = best_it->second.vpl * 2.0 * scale_factor;
            
            marker.color.a = 0.5; // Semi-transparent
            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            
            pub_marker_.publish(marker);
        }
        ROS_INFO("Published PL at %.4f (data time: %.4f)", stamp.toSec(), best_it->first);
    }

    void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
        publishForTimestamp(msg->header.stamp, msg->header.frame_id, msg->child_frame_id);
    }

    const std::map<double, IntegrityRecord>& getData() const { return data_map_; }
    ros::Publisher& getPublisher() { return pub_pl_; }

private:
    ros::Publisher pub_pl_;
    ros::Publisher pub_marker_;
    std::map<double, IntegrityRecord> data_map_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "publish_integrity_log");
    ros::NodeHandle nh("~");

    std::string file_path;
    std::string topic_name;
    std::string trigger_topic;
    std::string frame_id;
    double speed_factor;
    
    nh.param<std::string>("file_path", file_path, "");
    nh.param<std::string>("topic_name", topic_name, "/gici/integrity/pl_replay");
    nh.param<std::string>("trigger_topic", trigger_topic, ""); // If set, syncs with this topic
    nh.param<std::string>("frame_id", frame_id, "map");
    nh.param<double>("speed_factor", speed_factor, 1.0);

    if (file_path.empty()) {
        ROS_ERROR("Parameter 'file_path' is required!");
        return -1;
    }

    IntegrityReplayer replayer(nh, file_path, topic_name);

    if (!trigger_topic.empty()) {
        ROS_INFO("Running in Trigger Mode. Syncing with: %s", trigger_topic.c_str());
        ros::Subscriber sub = nh.subscribe(trigger_topic, 100, &IntegrityReplayer::odomCallback, &replayer);
        ros::spin();
    } else {
        ROS_INFO("Running in Timer Mode (Playback).");
        const auto& data = replayer.getData();
        if (data.empty()) return 0;

        auto it = data.begin();
        double prev_timestamp = it->first;
        
        while (ros::ok() && it != data.end()) {
            double timestamp = it->first;
            
            // Time sync
            double diff = timestamp - prev_timestamp;
            if (diff > 0) {
                ros::Duration(diff / speed_factor).sleep();
            }
            prev_timestamp = timestamp;

            replayer.publishForTimestamp(ros::Time(timestamp), frame_id);
            
            ROS_INFO_THROTTLE(1.0, "Published PL at %.4f", timestamp);
            
            it++;
            ros::spinOnce();
        }
    }

    return 0;
}
