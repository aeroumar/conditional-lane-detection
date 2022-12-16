#include <cmath>
#include <vector>
#include <iostream>
#include "ros/ros.h"
#include <algorithm>
#include <unordered_map>
#include <opencv2/video/tracking.hpp>

#include <aerovect_msgs/LaneMetadata.h>
#include <aerovect_msgs/LaneOffset.h>

#include <Eigen/QR>
#include <Eigen/Dense>

namespace perception
{
class LaneLateralOffset {
  public:
    LaneLateralOffset(const ros::NodeHandle &nh) : nh_(nh) {
      int img_width = 1280;
      int img_height = 720;

      ros::Rate ros_rate_ = new ros::Rate(10);

      aerovect_msgs::LaneMetadata curr_lanes_metadata_msg;
      bool new_msg_received = false;
      // subscriber to inference metadata topic
      ros::Subscriber lane_metadata_sub_ = nh_.subscribe(
        "/lane_detection/lane_metadata", 5, &lane_metadata_clk_, this);
      // publisher to lateral offset topic
      ros::Publisher lateral_offset_pub_ = nh_.advertise<aerovect_msgs::LaneOffset>(
        "/lane_detection/lateral_offset", 5);     
    }

    bool param_server() {
      const auto node_name = ros::this_node::getName();
      (void)node_name;  // For param server lookups (non-private)
      bool lookup_fail = true; 
      bool loose_params;
      if (!nh_.param<bool>("loose_params", loose_params, true)) {
        ROS_WARN("ROS param 'loose_params' not set");
        lookup_fail = false;
      }
      return lookup_fail || loose_params;
    }

    void reshape_lanes_(const aerovect_msgs::LaneMetadata &lanes, std::vector<std::vector<int>> reshaped_lanes) {
      std::vector<int> temp_;
      for (int i = 0; i < lanes.lane_instances.size(); i++) {
        temp_.push_back(lanes.lane_instances.at(i));
        if (temp_.size() == lanes.h_samples.size()) {
          reshaped_lanes.push_back(temp_);
          temp_.clear();
        }
      }
    }

    void lane_metadata_clk_(const aerovect_msgs::LaneMetadata &lanes) {
      ROS_INFO_ONCE("Subscribed to lane metadata");
      if (nullptr != lanes) {
        curr_lanes_metadata_msg = lanes;
        new_msg_received = true;
      }
    }

    void get_ego_lane_(
      const std::vector<std::vector<int>> &lanes,
      const std::vector<int> &h_samples, 
      std::unordered_map<std::string, std::vector<int>> &left_lane, 
      std::unordered_map<std::string, std::vector<int>> &right_lane, 
      std::vector<int> &left_lane_idx, 
      std::vector<int> &right_lane_idx
      ) {
      int idx = 0;
      left_lane_idx = {};
      right_lane_idx = {};
      int mid_x = img_width/2;
      int smallest_left_dist = 50000;
      int smallest_right_dist = 50000;

      for (std::vector<int> lane : lanes) {
        int count = 0;
        // iterating backwards over lane x-values
        for (auto it = lane.rbegin(); it != lane.rend(); ++it) {
          int curr_pix = *it;
          if (curr_pix == -2) {
            count++;
            continue;
          }
          else {
            int curr_lane_x_dist = curr_pix - mid_x;
            if (curr_lane_x_dist < 0) {
              if (left_lane_idx.empty()) {
                left_lane_idx.push_back(idx);
              }
              else if (abs(curr_lane_x_dist) < smallest_left_dist) {
                left_lane_idx.at(0) = idx;
              }
            }
            else if (curr_lane_x_dist > 0) {
              if (right_lane_idx.empty()) {
                right_lane_idx.push_back(idx);
              }
              else if (abs(curr_lane_x_dist) < smallest_right_dist) {
                right_lane_idx.at(0) = idx;
              }
            }
            idx++;
            break;
          }
        }
      }
      if (left_lane_idx.empty() || left_lane_idx.empty()) {
        return;
      }
      left_lane = {{"x", {}}, {"y", {}}};
      right_lane = {{"x", {}}, {"y", {}}};
      int count = 0;
      for (int lane_x : lanes.at(left_lane_idx.at(0))) {\
        // std::cout << lane_x << std::endl;
        if (lane_x == -2) {
          count++;
          continue;
        }
        left_lane["x"].push_back(lane_x);
        left_lane["y"].push_back(h_samples.at(count));
        count++;
      }
      int count2 = 0;
      for (int lane_x : lanes.at(right_lane_idx.at(0))) {
        // std::cout << lane_x << std::endl;
        if (lane_x == -2) {
          count2++;
          continue;
        }
        right_lane["x"].push_back(lane_x);
        right_lane["y"].push_back(h_samples.at(count2));
        count2++;
      }     
      return;
    }

    void get_mid_of_lane_(
      const std::unordered_map<std::string, std::vector<int>> &ego_lane_left,
      const std::unordered_map<std::string, std::vector<int>> &ego_lane_right,
      std::vector<int> &smallest_det_lane_y,
      std::vector<std::pair<int, int>> &mid_curve
    ) {
      mid_curve = {};
      if (ego_lane_left["x"].size() <= ego_lane_right["x"].size()) {
        smallest_det_lane_y = ego_lane_left["y"];
        for (int i = 0; i < ego_lane_left["x"].size(); i++) {
          mid_curve.push_back(std::make_pair(floor((ego_lane_right["x"].at(i)+ego_lane_left["x"].at(i))/2), ego_lane_left["y"].at(i)));
        }
      }
      else if (ego_lane_left["x"].size() > ego_lane_right["x"].size()) {
        smallest_det_lane_y = ego_lane_right["y"];
        for (int i = 0; i < ego_lane_right["x"].size(); i++) {
          mid_curve.push_back(std::make_pair(floor((ego_lane_right["x"].at(i)+ego_lane_left["x"].at(i))/2), ego_lane_right["y"].at(i)));
        }
      }
      return;
    }

    void get_perspective_transform_(
      const std::vector<cv::Point2f> &src_pts,
      const std::vector<cv::Point2f> &dst_pts,
      cv::Mat_<float> &M
    ) {
      M = cv::getPerspectiveTransform(src_pts, dst_pts);
    }

    void get_inverse_perspective_(
      const std::vector<cv::Point2f> &src_pts,
      const std::vector<cv::Point2f> &dst_pts,
      const std::vector<std::pair<int, int>> &transformed_offset,
      std::vector<std::pair<int, int>> &normal_offset      
    ) {
      std::vector<cv::Point2f> src;

      cv::Mat_<float> M = cv::getPerspectiveTransform(dst_pts, src_pts);
      for (int i = 0; i < transformed_offset.size(); i++)
      {
        std::pair<int,int> point_ = transformed_offset.at(i);
        std::vector<cv::Point2f> cv_point_ = {cv::Point2f(point_.first, point_.second)};
        std::vector<cv::Point2f> transformed_point_;
        cv::perspectiveTransform(cv_point_, transformed_point_, M);
        normal_offset.push_back(std::make_pair(transformed_point_.at(0).x, transformed_point_.at(0).y));
      }
    }

    void get_transformed_points_(
      const std::vector<std::pair<int, int>> &points,
      const cv::Mat_<float> &matrix,  
      std::vector<float> &transformed_offset_x,
      std::vector<float> &transformed_offset_y,
      std::vector<std::pair<int, int>> &transformed_offset,
      bool &good_perp_transform,       
      int offset_px = 0    
    ) {
      for (int i = 0; i < points.size(); i++) {
        std::pair<int, int> point = points.at(i);
        float row1 = matrix(0,0)*point.first + matrix(0,1)*point.second + matrix(0,2);
        float row2 = matrix(1,0)*point.first + matrix(1,1)*point.second + matrix(1,2);
        float row3 = matrix(2,0)*point.first + matrix(2,1)*point.second + matrix(2,2);
        if (abs(row3) <= 0.0001 or abs(row3) == 0.0001) {
          transformed_offset.push_back(std::make_pair(0,0));
          transformed_offset_x.push_back(float(0));
          transformed_offset_y.push_back(float(0));
          good_perp_transform = false;
        }
        else {
          float px = row1/row3;
          float py = row2/row3;
          float x_offset = px-offset_px;
          transformed_offset.push_back(std::make_pair(int(x_offset),int(py)));
          transformed_offset_x.push_back(x_offset);
          transformed_offset_y.push_back(py);
          good_perp_transform = true;          
        }
      }      
    }

    void map_to_vect_coords_(
      const std::unordered_map<std::string, std::vector<int>> &input,
      std::vector<std::pair<int, int>> &output
    ) {
      std::cout << "input: " << input["x"][0] << "," << input["y"][0] << std::endl;
      // iterate over vector
      for (int i = 0; i < input["x"].size(); i++) {
        output.push_back(std::make_pair(input["x"].at(i), input["y"].at(i)));
      }
      std::cout << "output: " << output.at(0).first << "," << output.at(0).second << std::endl;
    }
    

    void polyfit_(	
      const std::vector<double> &t,
      const std::vector<double> &v,
      std::vector<double> &coeff,
      int order
    ) {
      // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
      Eigen::MatrixXd T(t.size(), order + 1);
      Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
      Eigen::VectorXd result;

      // check to make sure inputs are correct
      assert(t.size() == v.size());
      assert(t.size() >= order + 1);
      // Populate the matrix
      for(size_t i = 0 ; i < t.size(); ++i)
      {
        for(size_t j = 0; j < order + 1; ++j)
        {
          T(i, j) = pow(t.at(i), j);
        }
      }
      // std::cout<<T<<std::endl;
      
      // Solve for linear least square fit
      result  = T.householderQr().solve(V);
      coeff.resize(order+1);
      for (int k = 0; k < order+1; k++)
      {
        coeff[k] = result[k];
      }
    }

    void find_curvature_(const std::vector<int> &x_pts, std::vector<double> &curvature) {
      for (int i = 0; i < x_pts.size(); i++) {
        int i_left = i - 1;
        int i_right = i + 1;
        if (i_left < 0) {
          i_left = 0;
          i_right = 1;
        }
        if (i_right >= x_pts.size()) {
          i_right = x_pts.size() - 1;
          i_left = i_right - 1;
        }
        // gradient value at i
        double dist_grad = (x_pts.at(i_right) - x_pts.at(i_left))/2.0;
        curvature.push_back(dist_grad);
      }
    }

    double get_lateral_offset(
      const std::vector<std::vector<int>> &lanes, 
      const std::vector<int> &h_samples
    ) {
      // TODO: Move filter definition into constructor so that it is reused.
      cv::KalmanFilter kalman_filter(4,3);

      float delta_t = 0.1;
      kalman_filter.measurementMatrix = (cv::Mat_<float>(3, 4) << 1,0,0,0, 0,1,0,0, 0,0,1,0); 
      kalman_filter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,delta_t,0,0, 0,1,0,0, 0,0,1,delta_t, 0,0,0,1);

      double offset_mtr = 0.0;
      int bad_frame_count = 0;
      bool spurious_frame = false;

      std::vector<int> ego_lane_left_idx; 
      std::vector<int> ego_lane_right_idx;
      std::unordered_map<std::string, std::vector<int>> ego_lane_left; 
      std::unordered_map<std::string, std::vector<int>> ego_lane_right;

      get_ego_lane_(lanes, h_samples, ego_lane_left, ego_lane_right, ego_lane_left_idx, ego_lane_right_idx);
      if (ego_lane_left_idx.empty() || ego_lane_right_idx.empty()) {
        // TODO: Return False or equivalent here
        return;
      }

      std::vector<int> smallest_det_lane_y;
      std::vector<std::pair<int, int>> mid_curve;
      get_mid_of_lane_(ego_lane_left, ego_lane_right, smallest_det_lane_y, mid_curve);

      std::vector<cv::Point2f> src_pts;
      src_pts.push_back(cv::Point2f(ego_lane_left["x"].at(0)-30, smallest_det_lane_y.at(0)));
      src_pts.push_back(cv::Point2f(ego_lane_right["x"].at(0)+30, smallest_det_lane_y.at(0)));
      // TODO: .back() errors out if the vector is empty so may need to do if (!vec.empty()) check before running this
      src_pts.push_back(cv::Point2f(ego_lane_right["x"].back(), ego_lane_right["y"].back()));
      src_pts.push_back(cv::Point2f(ego_lane_left["x"].back(), ego_lane_left["y"].back()));

      std::vector<cv::Point2f> dst_pts;
      dst_pts.push_back(cv::Point2f(150,10));
      dst_pts.push_back(cv::Point2f(1200,10));
      dst_pts.push_back(cv::Point2f(1200,700));
      dst_pts.push_back(cv::Point2f(150,700));

      cv::Mat_<float> transform_matrix;
      get_perspective_transform_(src_pts, dst_pts, transform_matrix);

      int offset_px = (mid_curve.back().first) - img_width/2;
      bool good_perp_transform; 
      // cam
      std::vector<float> transformed_cam_x;
      std::vector<float> transformed_cam_y;
      std::vector<std::pair<int, int>> transformed_cam;
      get_transformed_points_(mid_curve, transform_matrix, transformed_cam_x, 
      transformed_cam_y, transformed_cam, good_perp_transform, offset_px);
      // mid
      std::vector<float> transformed_mid_x;
      std::vector<float> transformed_mid_y;   
      std::vector<std::pair<int, int>> transformed_mid;
      get_transformed_points_(mid_curve, transform_matrix, transformed_mid_x, 
      transformed_mid_y, transformed_mid, good_perp_transform);  
      // left lane
      std::vector<float> transformed_left_x;
      std::vector<float> transformed_left_y;   
      std::vector<std::pair<int, int>> ego_left_lane_list;
      std::vector<std::pair<int, int>> transformed_left;
      map_to_vect_coords_(ego_lane_left, ego_left_lane_list);
      get_transformed_points_(ego_left_lane_list, transform_matrix, transformed_left_x, 
      transformed_left_y, transformed_left, good_perp_transform);  
      // right lane
      std::vector<float> transformed_right_x;
      std::vector<float> transformed_right_y;   
      std::vector<std::pair<int, int>> ego_right_lane_list;
      std::vector<std::pair<int, int>> transformed_right;      
      map_to_vect_coords_(ego_lane_right, ego_right_lane_list); 
      get_transformed_points_(ego_right_lane_list, transform_matrix, transformed_right_x, 
      transformed_right_y, transformed_right, good_perp_transform);     
      
      bool reliable_measurement = false;
      double filtered_offset_mtr;
      if (good_perp_transform) {
        std::vector<std::pair<int, int>> perspective_cam;
        get_inverse_perspective_(src_pts, dst_pts, transformed_cam, perspective_cam);
        std::cout << "transformed_mid_y: " << std::endl;
        for (auto i : transformed_mid_y) {
          std::cout << i << std::endl;
        }
        std::vector<double> mid_curve_coeffs;
        // try swapping the x,y vectors below if results look off
        std::vector<double> transformed_mid_x_double(transformed_mid_x.begin(), transformed_mid_x.end());
        std::vector<double> transformed_mid_y_double(transformed_mid_y.begin(), transformed_mid_y.end());
        polyfit_(transformed_mid_x_double, transformed_mid_y_double, mid_curve_coeffs, 2);    

        std::vector<double> normal_off_coeffs;
        // try swapping the x,y vectors below if results look off
        std::vector<double> transformed_cam_x_double(transformed_cam_x.begin(), transformed_cam_x.end());
        std::vector<double> transformed_cam_y_double(transformed_cam_y.begin(), transformed_cam_y.end());
        polyfit_(transformed_cam_x_double, transformed_cam_y_double, normal_off_coeffs, 2);

        // int min_lane_pts = min(transformed_left_y.size(), transformed_right_y.size());
        // float lane_width_pix_list = 0;

        // for (int = 0; i < min_lane_pts; i++) {
        //   lane_width_pix_list += abs(transformed_left_x.at(i) - transf ormed_right_x.at(i))
        // }
        double mid_curve_y = pow(mid_curve_coeffs.at(0)*img_width,2) + mid_curve_coeffs.at(1)*img_width + mid_curve_coeffs.at(2);
        double normal_off_y = pow(normal_off_coeffs.at(0)*img_width,2) + normal_off_coeffs.at(1)*img_width + normal_off_coeffs.at(2);
        
        int offset_px1 = perspective_cam.back().first - mid_curve.back().first;
        double offset_px_coeff = normal_off_y - mid_curve_y;

        double offset_mtr_prev = offset_mtr;
        offset_mtr = offset_px1*0.005;
        if (abs(offset_mtr_prev - offset_mtr) > 0.7) {
          offset_mtr = offset_mtr_prev;
          bad_frame_count++; 
        }
        else {
          bad_frame_count = 0;
        }

        if (bad_frame_count >= 3) {
          reliable_measurement = false;
        }
        else {
          reliable_measurement = true;
        }        
        
        if (ego_lane_left["x"].size() > 3 && ego_lane_right["x"].size() > 3) {
          std::vector<double> curvature_left;
          std::vector<double> curvature_right;
          find_curvature_(ego_lane_left["x"], curvature_left);
          find_curvature_(ego_lane_right["x"], curvature_right);

          std::vector<int> left_curvature_check;
          std::vector<int> right_curvature_check;
          for (int i = 0; i < curvature_left.size(); i++) {
            if (curvature_left.at(i) >= 110) {
              left_curvature_check.push_back(i);
            }
          }
          for (int i = 0; i < curvature_right.size(); i++) {
            if (curvature_right.at(i) >= 110) {
              right_curvature_check.push_back(i);
            }
          }
          if (left_curvature_check.size() >= 2 || right_curvature_check.size() >= 2) {
            spurious_frame = true;
          }
          else {
            spurious_frame = false;
          }
          std::cout << "left_curvature_check: " << std::endl;
          for (auto i : left_curvature_check) {
            std::cout << i << std::endl;
          }    
        }
        float y_dot = 0.0;
        float yaw = 0.0;

        cv::Mat prediction = kalman_filter.predict();

        float measurement_[] = {offset_mtr, y_dot, yaw};
        cv::Mat measurement(3, 1, CV_32F, measurement_);
        kalman_filter.correct(measurement);
        filtered_offset_mtr = kalman_filter.statePost.at<float>(0);

        std::cout << "measurement: " << measurement << std::endl; 
        std::cout << "kalman_filter.statePost: " << kalman_filter.statePost << std::endl;        
        std::cout << "filtered_offset_mtr: " << filtered_offset_mtr << std::endl;        
      }
      return filtered_offset_mtr;
    }

    void offset_calc_loop() {
      if (new_msg_received) {
        std::vector<std::vector<int>> reshaped_lanes;
        reshape_lanes_(curr_lanes_metadata_msg, reshaped_lanes);
        aerovect_msgs::LaneOffset offset_msg;
        offset_msg.header.frame_id = curr_lanes_metadata_msg->header.frame_id;
        offset_msg.header.stamp = curr_lanes_metadata_msg->header.stamp;
        offset_msg.offset = offset_calculator.get_lateral_offset(reshaped_lanes, curr_lanes_metadata_msg.h_samples);
        lateral_offset_pub_.publish(offset_msg);
        new_msg_received = false;
      }
      ros::spinOnce();
      ros_rate_->sleep();      
    }

    void run() {
      while (ros::ok()) {
        ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.1));
        offset_calc_loop();
      }
    }
};
} // namespace perception

int main(int argc, char **argv) {
  // Sample Input:
  std::vector<std::vector<int>> lanes = {
    {-2, -2, -2, 535, 496, 458, 427, 394, 366, 337, 307, 279, 250, 221, 191, 163, 135, 106, 78, 48, 22, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2},
    {-2, -2, -2, 649, 660, 669, 678, 689, 698, 707, 717, 726, 734, 744, 752, 761, 770, 779, 788, 797, 804, 814, 822, 830, 840, 847, 856, 865, 873, 882, 891, 898, 907, 915, 925, 942, 963, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2},
    {-2, -2, -2, 589, 578, 567, 556, 544, 533, 522, 512, 500, 489, 479, 468, 457, 447, 436, 426, 416, 406, 395, 385, 375, 364, 354, 344, 334, 323, 312, 302, 292, 283, 272, 262, 252, 242, 231, 222, 212, 202, 192, 182, 172, 163, 154, 144, 133}
  };
  std::vector<int> h_samples = {240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710};
  
  // init ros
  ros::init(argc, argv, "lane_lateral_offset");  
  ros::NodeHandle nh_("~");

  perception::LaneLateralOffset offset_calculator(nh_);
  if (offset_calculator.param_server()) {
    offset_calculator.run();
  }  
  return 0;
}
