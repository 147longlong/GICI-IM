/**
* @Function: Integrity Support Message Generation for Visual Part
* 
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
* 
* Copyright (C) 2025 by Yulong Sun, All rights reserved.
**/

#include "gici/integrity/visual_ism_gen.h"
#include "gici/utility/transform.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace gici {

VisualISMGenerator::VisualISMGenerator(const VisualISMOptions& ism_options, const DetectorOptions& options, const CameraPtr& cam) 
      : options_(ism_options), grid_(options.cell_size, std::ceil(static_cast<double>(cam->imageWidth())/options.cell_size),
              std::ceil(static_cast<double>(cam->imageHeight())/options.cell_size)) {

    img_count = 0;

    if (!options_.saved_file.empty()) {
      std::ofstream out(options_.saved_file, std::ios::trunc);
      out << "#frame_id all_couts outlier_count" << std::endl;
    }

    if (!options_.sd_errors_file.empty()) {
      std::ofstream out_sd(options_.sd_errors_file, std::ios::trunc);
      out_sd << "#sampson_error" << std::endl;
    }

    if (!options_.pairwise_file.empty()) {
      std::ios_base::openmode mode = std::ios::out;
      if (options_.pairwise_file_mode == "app") {
        mode |= std::ios::app;
      } else {
        mode |= std::ios::trunc;
      }

      std::ofstream out_pairwise(options_.pairwise_file, mode);
      out_pairwise << "# PairType Error" << std::endl;
      
    }

    if (!options_.gt_file.empty()) {
      loadGTData();
    }
}

void VisualISMGenerator::loadGTData() {
    std::ifstream in(options_.gt_file);
    if (!in.is_open()) {
        LOG(ERROR) << "Cannot open GT file: " << options_.gt_file;
        CHECK_EQ(1, 0) << "Please provide a valid GT file. When the ISM generation is enabled, GT file is required.";
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        GTPose pose;
        iss >> pose.time >> pose.pos[0] >> pose.pos[1] >> pose.pos[2] 
            >> pose.q.coeffs()[0] >> pose.q.coeffs()[1] 
            >> pose.q.coeffs()[2] >> pose.q.coeffs()[3];
        gt_poses_.push_back(pose);
    }
    std::sort(gt_poses_.begin(), gt_poses_.end(), [](const GTPose& a, const GTPose& b) {
        return a.time < b.time;
    });
    LOG(INFO) << "Loaded " << gt_poses_.size() << " GT poses.";
}

bool VisualISMGenerator::getGTPose(double timestamp, Eigen::Vector3d& pos, Eigen::Quaterniond& q) {
    if (gt_poses_.empty()) return false;
    
    double target_time = timestamp + 18.0;
    
    auto it = std::lower_bound(gt_poses_.begin(), gt_poses_.end(), target_time, 
        [](const GTPose& pose, double val) {
            return pose.time < val;
        });
        
    if (it != gt_poses_.end()) {
        if (std::abs(it->time - target_time) < 0.005) {
            pos = it->pos;
            q = it->q;
            return true;
        }
    }
    if (it != gt_poses_.begin()) {
        auto prev = std::prev(it);
        if (std::abs(prev->time - target_time) < 0.005) {
            pos = prev->pos;
            q = prev->q;
            return true;
        }
    }
    return false;
}

bool VisualISMGenerator::getGTFundamentalMat(Eigen::Quaterniond& q_cur_ref, Eigen::Vector3d& pos_cur_ref){
    
    Eigen::Vector3d cur_pos, ref_pos;
    Eigen::Quaterniond cur_q, ref_q;
    
    if (!getGTPose(cur_frame_->getTimestampSec(), cur_pos, cur_q)) {
        return false;
    }
    if (!getGTPose(ref_frame_->getTimestampSec(), ref_pos, ref_q)) {
        std::cout << "Cannot find the reference frame" << std::endl;
        return false;
    }

    std::cout << "*********************get GT fundamentalMat************************" << std::endl;
    std::cout << "Cur Frame Time: " << cur_frame_->getTimestampSec() << std::endl;
    std::cout << "Position: " << cur_pos.transpose() << std::endl;
    std::cout << "Orientation: " << cur_q.coeffs().transpose() << std::endl;
  
    std::cout << "Ref Frame Time: " << ref_frame_->getTimestampSec() << std::endl;
    std::cout << "Position: " << ref_pos.transpose() << std::endl;
    std::cout << "Orientation: " << ref_q.coeffs().transpose() << std::endl;
  
    // Normalize cur_q
    if (std::abs(cur_q.squaredNorm() - 1.0) > 1e-6) {
      cur_q.normalize();
    }
  
    // Normalize ref_q
    if (std::abs(ref_q.squaredNorm() - 1.0) > 1e-6) {
      ref_q.normalize();
    }
    if (!std::isfinite(cur_q.squaredNorm()) || !std::isfinite(ref_q.squaredNorm())) {
      return false;
    }
    Transformation T_cur_gt = Transformation(cur_pos, cur_q); // tum is body to world
    Transformation T_ref_gt = Transformation(ref_pos, ref_q); 
    Transformation T_WS_cam_cur = (T_cur_gt * cur_frame_->T_imu_cam()); 
    Transformation T_WS_cam_ref = (T_ref_gt * ref_frame_->T_imu_cam());
  
    std::cout << "T_WS_cam_ref: " << T_WS_cam_ref.getEigenQuaternion().coeffs().transpose() << " " << T_WS_cam_ref.getPosition().transpose() << std::endl;
    std::cout << "T_WS_cam_cur: " << T_WS_cam_cur.getEigenQuaternion().coeffs().transpose() <<  " " << T_WS_cam_cur.getPosition().transpose() << std::endl;
    // Compute relative pose q_cur_ref is ref to cur
    Transformation T_cur_ref = T_WS_cam_cur.inverse() * T_WS_cam_ref;
    q_cur_ref = T_cur_ref.getEigenQuaternion();
    pos_cur_ref = T_cur_ref.getPosition();
    if (std::abs(q_cur_ref.squaredNorm() - 1.0) > 1e-6) {
      std::cout << "Warning: q_cur_ref is not normalized. Normalizing manually." << std::endl;
      q_cur_ref.normalize();
    }   
  
    return true;
}


bool VisualISMGenerator::getGTFundamentalMat(const FramePtr& f1, const FramePtr& f2, Eigen::Quaterniond& q_1_2, Eigen::Vector3d& pos_1_2){
    
    Eigen::Vector3d pos1, pos2;
    Eigen::Quaterniond q1, q2;
    
    if (!getGTPose(f1->getTimestampSec(), pos1, q1)) return false;
    if (!getGTPose(f2->getTimestampSec(), pos2, q2)) return false;

    // Normalize q1
    if (std::abs(q1.squaredNorm() - 1.0) > 1e-6) {
      q1.normalize();
    }
  
    // Normalize q2
    if (std::abs(q2.squaredNorm() - 1.0) > 1e-6) {
      q2.normalize();
    }
    
    Transformation T_1_gt = Transformation(pos1, q1); 
    Transformation T_2_gt = Transformation(pos2, q2); 
    Transformation T_WS_cam_1 = (T_1_gt * f1->T_imu_cam()); 
    Transformation T_WS_cam_2 = (T_2_gt * f2->T_imu_cam());
  
    // Compute relative pose q_1_2 is f2 to f1
    Transformation T_1_2 = T_WS_cam_1.inverse() * T_WS_cam_2;
    q_1_2 = T_1_2.getEigenQuaternion();
    pos_1_2 = T_1_2.getPosition();
    if (std::abs(q_1_2.squaredNorm() - 1.0) > 1e-6) {
      q_1_2.normalize();
    }   
  
    return true;
}


void VisualISMGenerator::saveRawTrackedImage() {

  // Save the tracked features
  std::string file = options_.saved_folder + std::to_string(img_count) + ".png";
  std::string file_combined = options_.saved_folder + std::to_string(img_count) + "_combined.png";

  cv::Mat img = cur_frame_->img_pyr_[0].clone();
  cv::Mat cur_img = cur_frame_->img_pyr_[0].clone();
  cv::Mat ref_img = ref_frame_->img_pyr_[0].clone();

  cv::cvtColor(ref_img, ref_img, cv::COLOR_GRAY2BGR); 
  cv::cvtColor(cur_img, cur_img, cv::COLOR_GRAY2BGR); 

  cv::Mat img_combined(std::max(ref_img.rows, cur_img.rows), ref_img.cols + cur_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  ref_img.copyTo(img_combined(cv::Rect(0, 0, ref_img.cols, ref_img.rows)));
  cur_img.copyTo(img_combined(cv::Rect(ref_img.cols, 0, cur_img.cols, cur_img.rows)));

  const cv::Mat& mask = cur_frame_->cam()->getMask();

  int all_couts = 0;
  for (int j = 0; j < index_map_.size(); j++) {

    int i = index_map_.at(j);

    if (ref_frame_->type_vec_[i] == FeatureType::kOutlier) continue;

    if(!mask.empty() && mask.at<uint8_t>(
      static_cast<int>(cur_points_[i].y), static_cast<int>(cur_points_[i].x)) == 0) {
      continue;
    }

    size_t grid_index = grid_.getCellIndex(cur_points_[i].x,cur_points_[i].y, 1);
    if (!grid_.isOccupied(grid_index)) {
      all_couts++;
      cv::circle(img, ref_points_[i], 1, cv::Scalar(128, 128, 255), 2); // ref_points_
      cv::circle(img, cur_points_[i], 1, cv::Scalar(0, 0, 255), 2); // cur_points_
      cv::line(img, ref_points_[i], cur_points_[i], cv::Scalar(0, 0, 255)); 

      cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
      cv::circle(img_combined, ref_points_[i], 2, random_color, -1);
      cv::Point2f cur_point_shifted = cur_points_[i] + cv::Point2f(static_cast<float>(ref_img.cols), 0);
      cv::circle(img_combined, cur_point_shifted, 2, random_color, -1);
      cv::line(img_combined, ref_points_[i], cur_point_shifted, random_color, 1);
      grid_.setOccupied(grid_index);
    }

  }
  grid_.reset();

  cv::imwrite(file, img);
  cv::imwrite(file_combined, img_combined);
  LOG(INFO) << "Saved raw tracked image: " << file;
  LOG(INFO) << "Saved combined image: " << file_combined;

}


void VisualISMGenerator::saveOutliers() {

  std::string img_file = options_.saved_folder + std::to_string(img_count) + "_exclude.png";

  cv::Mat cur_img = cur_frame_->img_pyr_[0].clone();
  cv::Mat ref_img = ref_frame_->img_pyr_[0].clone();
  cv::cvtColor(ref_img, ref_img, cv::COLOR_GRAY2BGR); 
  cv::cvtColor(cur_img, cur_img, cv::COLOR_GRAY2BGR); 

  cv::Mat img_combined(std::max(ref_img.rows, cur_img.rows), ref_img.cols + cur_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  ref_img.copyTo(img_combined(cv::Rect(0, 0, ref_img.cols, ref_img.rows)));
  cur_img.copyTo(img_combined(cv::Rect(ref_img.cols, 0, cur_img.cols, cur_img.rows)));
  cv::Mat img_exclude = img_combined.clone();

  const cv::Mat& mask = cur_frame_->cam()->getMask();

  outlier_count = 0;

  for (int i = 0; i < index_map_.size(); i++) {

    int j = index_map_.at(i);

    if (ref_frame_->type_vec_[j] == FeatureType::kOutlier) continue;

    if(!mask.empty() && mask.at<uint8_t>(
      static_cast<int>(cur_points_[j].y), static_cast<int>(cur_points_[j].x)) == 0) {
      continue;
    }
    size_t grid_index = grid_.getCellIndex(cur_points_[j].x,cur_points_[j].y, 1);
    if (!grid_.isOccupied(grid_index)) {
      if (!status_[i]){
        outlier_count++;
        cv::circle(img_exclude, ref_points_[j], 2, cv::Scalar(0, 0, 255), -1);
        cv::Point2f cur_point_shifted = cur_points_[j] + cv::Point2f(static_cast<float>(ref_img.cols), 0);
        cv::circle(img_exclude, cur_point_shifted, 2, cv::Scalar(0, 0, 255), -1);
        cv::line(img_exclude, ref_points_[j], cur_point_shifted, cv::Scalar(0, 0, 255), 1);
      }
      grid_.setOccupied(grid_index);
    }
  }
  grid_.reset();

  cv::imwrite(img_file, img_exclude);
  LOG(INFO) << "Saved outliers image: " << img_file;
}


void VisualISMGenerator::saveInliers() {

  std::string file_ransac = options_.saved_folder + std::to_string(img_count) + "_ransac.png";
    
  // Save the tracked features
  cv::Mat cur_img_ransac = cur_frame_->img_pyr_[0].clone();
  cv::Mat ref_img_ransac = ref_frame_->img_pyr_[0].clone();
  cv::cvtColor(ref_img_ransac, ref_img_ransac, cv::COLOR_GRAY2BGR); 
  cv::cvtColor(cur_img_ransac, cur_img_ransac, cv::COLOR_GRAY2BGR); 
  cv::Mat img_combined_ransac(std::max(ref_img_ransac.rows, cur_img_ransac.rows), ref_img_ransac.cols + cur_img_ransac.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  ref_img_ransac.copyTo(img_combined_ransac(cv::Rect(0, 0, ref_img_ransac.cols, ref_img_ransac.rows)));
  cur_img_ransac.copyTo(img_combined_ransac(cv::Rect(ref_img_ransac.cols, 0, cur_img_ransac.cols, cur_img_ransac.rows)));
 
  const cv::Mat& mask = cur_frame_->cam()->getMask();

  for (int i = 0; i < index_map_.size(); i++) {
    if(!status_[i]) continue;

    int j = index_map_.at(i);

    if (ref_frame_->type_vec_[j] == FeatureType::kOutlier) continue;

    if(!mask.empty() && mask.at<uint8_t>(
      static_cast<int>(cur_points_[j].y), static_cast<int>(cur_points_[j].x)) == 0) {
      continue;
    }

    size_t grid_index = grid_.getCellIndex(cur_points_[j].x,cur_points_[j].y, 1);
    if (!grid_.isOccupied(grid_index)) {
      cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
      cv::circle(img_combined_ransac, ref_points_[j], 2, random_color, -1);
      cv::Point2f cur_point_shifted = cur_points_[j] + cv::Point2f(static_cast<float>(ref_img_ransac.cols), 0);
      cv::circle(img_combined_ransac, cur_point_shifted, 2, random_color, -1);
      cv::line(img_combined_ransac, ref_points_[j], cur_point_shifted, random_color, 1);
      grid_.setOccupied(grid_index);
    }
  }
  grid_.reset();

  cv::imwrite(file_ransac, img_combined_ransac);
  is_drawInliers = true;
  LOG(INFO) << "Saved inliers image: " << file_ransac;

}


void VisualISMGenerator::computePairwiseSampsonDistance(const FramePtr& cur_frame) {
    std::ofstream out_sd(options_.pairwise_file, std::ios::app);
    
    // Cache for relative poses and Essential matrices to avoid redundant calculations
    struct RelativePoseResult {
        bool valid;
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        Eigen::Matrix3d E;
    };
    std::map<std::pair<int, int>, RelativePoseResult> pose_cache;
    std::stringstream ss; // Buffer for file output

    // Iterate over all features in the current frame
    for (size_t i = 0; i < cur_frame->num_features_; ++i) {
        if (cur_frame->landmark_vec_[i] == nullptr) continue;
        
        PointPtr point = cur_frame->landmark_vec_[i];
        
        // Get observations
        std::vector<KeypointIdentifier> obs = point->obs_;
        
        // Sort observations by frame ID
        std::sort(obs.begin(), obs.end(), [](const KeypointIdentifier& a, const KeypointIdentifier& b) {
            return a.frame_id < b.frame_id;
        });
        
        // We need at least 2 observations to form a pair
        if (obs.size() < 2) continue;

        
        // Iterate over all previous observations to pair with current frame
        for (size_t j = 0; j < obs.size(); ++j) {
            if (obs[j].frame_id == cur_frame->id_) continue;

            FramePtr frame_j = obs[j].frame.lock();
            FramePtr frame_k = cur_frame;
            
            if (!frame_j) continue;
            
            // Check cache first
            std::pair<int, int> key = {frame_k->id_, frame_j->id_};
            RelativePoseResult res;
            
            auto it = pose_cache.find(key);
            if (it != pose_cache.end()) {
                res = it->second;
            } else {
                // Not in cache, compute and store
                res.valid = getGTFundamentalMat(frame_k, frame_j, res.q, res.t);
                if (res.valid) {
                    res.E = skewSymmetric(res.t) * res.q.toRotationMatrix();
                }
                pose_cache[key] = res;
            }
            
            if (!res.valid) continue; 

            // Get bearings
            BearingVector f_j = frame_j->f_vec_.col(obs[j].keypoint_index_);
            BearingVector f_k = frame_k->f_vec_.col(obs.back().keypoint_index_); 
            
            // Use cached Essential Matrix
            Eigen::Matrix3d E = res.E;
            
            cv::Mat f_j_norm = (cv::Mat_<double>(3, 1) << f_j.x()/f_j.z(), f_j.y()/f_j.z(), 1.0);
            cv::Mat f_k_norm = (cv::Mat_<double>(3, 1) << f_k.x()/f_k.z(), f_k.y()/f_k.z(), 1.0);
            
            cv::Mat E_cv = (cv::Mat_<double>(3, 3) << E(0, 0), E(0, 1), E(0, 2), E(1, 0), E(1, 1), E(1, 2), E(2, 0), E(2, 1), E(2, 2));
            
            double sd_dist_norm = cv::sampsonDistance(f_j_norm, f_k_norm, E_cv);
            
            // Convert to pixel units using focal length approximation
            // errorMultiplier() returns roughly f_x
            double focal_length = frame_k->cam()->errorMultiplier(); 
            double sd_dist_pixel2 = sd_dist_norm * focal_length * focal_length; // the sampson distance scales with the square of the focal length

            ss << j << "-" << obs.size() - 1 << " " << sd_dist_pixel2 << "\n";

            // LOG(INFO) << "[Visual ISM]: Pairwise SD between Frame " << frame_j->id_ 
            //           << " and Frame " << frame_k->id_ << " (" << j << "-" << obs.size() - 1 << ") : " << sd_dist_pixel << " pixels.";
        }
    }
    LOG(INFO) << "[Visual ISM]: Saved pairwise Sampson distances for Frame " << cur_frame->id_;
    out_sd << ss.str();
}

void VisualISMGenerator::computeSampsonDistance() {
  // Compute the Sampson distance
  Eigen::Quaterniond q_cur_ref_gt;
  Eigen::Vector3d pos_cur_ref_gt;
  if(!getGTFundamentalMat(q_cur_ref_gt, pos_cur_ref_gt)){
    return;
  }
  // std::cout << "Relative Pose: " << q_cur_ref_gt.coeffs().transpose() << " " << pos_cur_ref_gt.transpose() << std::endl;

  sampson_errors_map.clear();

  std::ofstream out_sd(options_.sd_errors_file, std::ios::app);

  const cv::Mat& mask = cur_frame_->cam()->getMask();

  for (int i = 0; i < index_map_.size(); i++) {
    if (!status_[i]) continue;

    int j = index_map_.at(i);

    if (ref_frame_->type_vec_[j] == FeatureType::kOutlier) continue;

    if(!mask.empty() && mask.at<uint8_t>(
      static_cast<int>(cur_points_[j].y), static_cast<int>(cur_points_[j].x)) == 0) {
      continue;
    }
    size_t grid_index = grid_.getCellIndex(cur_points_[j].x,cur_points_[j].y, 1);
    if (!grid_.isOccupied(grid_index)) {

      Eigen::Vector2d px_ref = Eigen::Vector2d(ref_points_[j].x, ref_points_[j].y);
      BearingVector bearing;
      ref_frame_->cam()->backProject3(px_ref, &bearing);
      cv::Mat f_ref = (cv::Mat_<double>(3, 1) << bearing.x() / bearing.z(), bearing.y() / bearing.z(), 1);
      Eigen::Vector2d px_cur = Eigen::Vector2d(cur_points_[j].x, cur_points_[j].y);
      cur_frame_->cam()->backProject3(px_cur, &bearing);
      cv::Mat f_cur = (cv::Mat_<double>(3, 1) << bearing.x() / bearing.z(), bearing.y() / bearing.z(), 1);

      Eigen::Matrix3d E = skewSymmetric(pos_cur_ref_gt) * q_cur_ref_gt.toRotationMatrix();
      cv::Mat E_cv = (cv::Mat_<double>(3, 3) << E(0, 0), E(0, 1), E(0, 2), E(1, 0), E(1, 1), E(1, 2), E(2, 0), E(2, 1), E(2, 2));
      
      double sd_dist = cv::sampsonDistance(f_ref, f_cur, E_cv);
      out_sd << sd_dist << std::endl;
      sampson_errors_map.insert(std::make_pair(j, sd_dist));
    }
  }
  
  grid_.reset();
}

void VisualISMGenerator::saveSDImage() {

  std::string file_ransac = options_.saved_folder + std::to_string(img_count) + "_ransac.png";
  if(!is_drawInliers){
    saveInliers();
    is_drawInliers = true;
  }

  cv::Mat img_combined_ransac = cv::imread(file_ransac);

  if (img_combined_ransac.empty()) {
    std::cout << "RANSAC image not found or cannot be read!" << std::endl;
    return;
  } 

  double mean_sampson = sampson_errors_map.size() > 0 ? std::accumulate(sampson_errors_map.begin(), sampson_errors_map.end(), 0.0, 
    [](double sum, const std::pair<int, double>& p) { return sum + p.second; }) / sampson_errors_map.size() : 1e-7;
  double max_sd = sampson_errors_map.size() > 0 ? std::max_element(sampson_errors_map.begin(), sampson_errors_map.end(), 
    [](const std::pair<int, double>& p1, const std::pair<int, double>& p2) { return p1.second < p2.second; })->second : 1e-7;
  int couts_fault = 0;

  for (const auto& pair : sampson_errors_map) {
    int index = pair.first; 
    double circle_radius = pair.second/mean_sampson * 2; 

    if(circle_radius > 40){
      circle_radius = 40;
    }
    if(circle_radius < 2){
      continue;
    }
    couts_fault++;
    cv::Point2f cur_point_shifted = cur_points_[index] + cv::Point2f(static_cast<float>(ref_frame_->img_pyr_[0].cols), 0);
    cv::circle(img_combined_ransac, cur_point_shifted, circle_radius, cv::Scalar(0, 0, 255), 1.5);
  
  }

  cv::imwrite(file_ransac, img_combined_ransac);
  // save outlier_count and all_couts to saved_file
  std::ofstream out(options_.saved_file, std::ios::app);
  out << img_count << " " << sampson_errors_map.size() << " " << outlier_count<< " mean_sd: " <<  mean_sampson  << " max_std: " << max_sd << " couts_fault: " << couts_fault << std::endl;
  LOG(INFO) << "Saved Sampson distance image: " << file_ransac;
}

void VisualISMGenerator::addImageCount() {
  img_count++;
  outlier_count = 0;
  is_drawInliers = false;
  sampson_errors_map.clear();
  grid_.reset();
  cur_frame_->clearFeatureStorage();
  ref_frame_->clearFeatureStorage();
  cur_points_.clear();
  ref_points_.clear();
  index_map_.clear();
  status_.clear();
}

}
