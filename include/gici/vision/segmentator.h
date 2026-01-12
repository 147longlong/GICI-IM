/**
* @Function: Lightweight image segmentation using MobileSAM and FastSAM
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025 by Yulong Sun, All rights reserved.
**/
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "gici/utility/common.h"
#include "gici/vision/python_segmentation_bridge.h"

namespace gici {

// Segmentation model type
enum class SegmentationModelType {
  FastSAM,    // Fast YOLOv8-based segmentation
  SLIC        // SLIC Superpixel Segmentation
};

// Segmentator options
struct SegmentatorOptions {
  // Model type to use
  SegmentationModelType model_type = SegmentationModelType::FastSAM;
  
  // Model paths (optional, will download if not provided)
  std::string fastsam_model_path = "";
  
  // Inference parameters
  bool use_gpu = false;               // Use GPU if available
  
};

// Segmentator class
class Segmentator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Segmentator(const SegmentatorOptions& options);
  ~Segmentator();
  
  // Segment image and return mask
  // Input: RGB image
  // Output: Binary mask (255 for foreground, 0 for background)
  cv::Mat segment(const cv::Mat& input);
  
  // Get visualization overlay
  cv::Mat getVisualization(const cv::Mat& input, const cv::Mat& mask);

  // Get class name from ID
  std::string getClassName(int class_id);
  
  // Warm up the model
  bool warmUp(const cv::Size& image_size = cv::Size(640, 480));
  
  // Check if model is loaded successfully
  bool isLoaded() const { return is_loaded_; }
  
  

private:
  SegmentatorOptions options_;
  bool is_loaded_;
  
  // Model-specific implementations
  void initializeFastSAM();
  void initializeSLIC();
  cv::Mat segmentFastSAM(const cv::Mat& input);
  cv::Mat segmentSLIC(const cv::Mat& input);
  
  // Model-specific data (using void* to avoid dependency issues)
  void* fastsam_model_;
  
  // Python bridge for PyTorch/ONNX inference
  std::shared_ptr<PythonSegmentationBridge> python_bridge_;
};

} // namespace gici