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
  MobileSAM,  // Lightweight SAM variant
  FastSAM,    // Fast YOLOv8-based segmentation
  OpenCV      // Traditional CV methods (GrabCut + Watershed)
};

// Segmentator options
struct SegmentatorOptions {
  // Model type to use
  SegmentationModelType model_type = SegmentationModelType::MobileSAM;
  
  // Model paths (optional, will download if not provided)
  std::string mobilesam_encoder_path = "";
  std::string mobilesam_decoder_path = "";
  std::string fastsam_model_path = "";
  
  // Inference parameters
  float confidence_threshold = 0.4f;  // Minimum confidence for detection
  float iou_threshold = 0.5f;         // IoU threshold for NMS
  bool use_gpu = false;               // Use GPU if available
  
  // OpenCV parameters (for traditional methods)
  int grabcut_iterations = 5;
  float watershed_threshold = 0.3f;
  
  // Performance settings
  int max_batch_size = 1;
  bool enable_cache = true;
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
  
  // Segment image with multiple prompts (for SAM-based models)
  // Input: RGB image, optional bounding boxes [N,4] in xyxy format
  // Output: Binary mask
  cv::Mat segmentWithPrompts(const cv::Mat& input, 
                            const std::vector<cv::Rect>& bboxes = {});
  
  // Get segmentation with instance masks (FastSAM only)
  // Returns vector of masks and their class IDs
  std::vector<std::pair<cv::Mat, int>> segmentInstances(const cv::Mat& input);

  // Get class name from ID
  std::string getClassName(int class_id);
  
  // Check if model is loaded successfully
  bool isLoaded() const { return is_loaded_; }
  
  // Get model type
  SegmentationModelType getModelType() const { return options_.model_type; }
  
  // Warm up the model (pre-load and run a dummy inference)
  bool warmUp(const cv::Size& image_size = cv::Size(640, 480));

private:
  SegmentatorOptions options_;
  bool is_loaded_;
  
  // Model-specific implementations
  void initializeMobileSAM();
  void initializeFastSAM();
  void initializeOpenCV();
  cv::Mat segmentMobileSAM(const cv::Mat& input);
  cv::Mat segmentFastSAM(const cv::Mat& input);
  cv::Mat segmentOpenCV(const cv::Mat& input);
  
  
  // Model-specific data (using void* to avoid dependency issues)
  void* mobilesam_model_;
  void* fastsam_model_;
  
  // Cache for model outputs
  std::map<std::string, cv::Mat> cache_;
  
  // Python bridge
  std::shared_ptr<PythonSegmentationBridge> python_bridge_;
};

} // namespace gici