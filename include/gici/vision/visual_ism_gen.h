/**
* @Function: Integrity Support Message Generation for Visual Part
* 
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
* 
* Copyright (C) 2025 by Yulong Sun, All rights reserved.
**/

#pragma once

#include "gici/utility/svo.h"

namespace gici {

class VisualISMGenerator {
public:
    // The constructor of visual_ism_gen_ is in feature_handler.cpp
    VisualISMGenerator(const DetectorOptions& options, const CameraPtr& cam);
    ~VisualISMGenerator() { }

    void setFrames(const FramePtr& cur_frame, const FramePtr& ref_frame, 
                  const std::unordered_map<int, int>& index_map, 
                  const std::vector<cv::Point2f>& cur_points, const std::vector<cv::Point2f>& ref_points) {
        cur_frame_ = cur_frame;
        ref_frame_ = ref_frame;

        index_map_ = index_map;
        cur_points_ = cur_points;
        ref_points_ = ref_points;
    }

    // get the relationship between current frame and reference frame after RANSAC 
    void setRANSACStatus(const std::vector<unsigned char>& status) {
        status_ = status;
    }

    // get the ground truth fundamental matrix between current frame and reference frame
    bool getGTFundamentalMat(Eigen::Quaterniond& q_cur_ref, Eigen::Vector3d& pos_cur_ref);

    // compute the Sampson distance for each feature point
    void computeSampsonDistance();

    // save the raw tracked image and combined two image with the features to the saved_folder
    void saveRawTrackedImage();
    
    // save the outliers image to the saved_folder with "count_exclude.png"
    void saveOutliers();

    // save the inliers image to the saved_folder with "count_ransac.png"
    void saveInliers();

    // save the Sampson distance image to the saved_folder with "count_ransac.png"
    void saveSDImage();
    
    void addImageCount();


private:
    std::string saved_folder;
    std::string saved_file;
    std::string gt_file;
    std::string sd_errors_file;

    FramePtr cur_frame_;
    FramePtr ref_frame_;
    OccupandyGrid2D grid_;
    std::unordered_map<int, int> index_map_;
    std::vector<cv::Point2f> cur_points_;
    std::vector<cv::Point2f> ref_points_;
    std::vector<unsigned char> status_;

    int img_count;
    int outlier_count = 0;

    std::ofstream out;
    std::ofstream out_sd;

    std::unordered_map<int, double> sampson_errors_map;
    bool is_drawInliers = false;

    

};

using VisualISMGeneratorPtr = std::shared_ptr<VisualISMGenerator>;

}