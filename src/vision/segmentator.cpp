/**
* @Function: Lightweight image segmentation implementation
*
* @Author  : GitHub Copilot
* @Email   : copilot@github.com
*
* Copyright (C) 2024 by GitHub Copilot, All rights reserved.
**/
#include "gici/vision/segmentator.h"
#include <fstream>

namespace gici {

// Constructor
Segmentator::Segmentator(const SegmentatorOptions& options) 
  : options_(options), is_loaded_(false), mobilesam_model_(nullptr), fastsam_model_(nullptr) {
  
  // Initialize based on model type
  switch (options_.model_type) {
    case SegmentationModelType::MobileSAM:
      initializeMobileSAM();
      break;
    case SegmentationModelType::FastSAM:
      initializeFastSAM();
      break;
    case SegmentationModelType::OpenCV:
      initializeOpenCV();
      break;
  }
}

// Destructor
Segmentator::~Segmentator() {
  // Clean up model resources
  if (mobilesam_model_) {
    // Cleanup MobileSAM resources
    mobilesam_model_ = nullptr;
  }
  if (fastsam_model_) {
    // Cleanup FastSAM resources
    fastsam_model_ = nullptr;
  }
}

// Initialize MobileSAM
void Segmentator::initializeMobileSAM() {
  LOG(INFO) << "Initializing MobileSAM...";
  
  std::string device = options_.use_gpu ? "cuda" : "cpu";
  std::string model_path = options_.mobilesam_encoder_path;
  if (model_path.empty()) {
      model_path = "mobile_sam.pt"; 
  }

  python_bridge_ = std::make_shared<PythonSegmentationBridge>();
  if (python_bridge_->initialize("mobile_sam", model_path, device)) {
    is_loaded_ = true;
    LOG(INFO) << "MobileSAM initialized successfully via Python bridge";
  } else {
    LOG(ERROR) << "Failed to initialize MobileSAM via Python bridge";
    is_loaded_ = false;
  }
}

// Initialize FastSAM
void Segmentator::initializeFastSAM() {
  LOG(INFO) << "Initializing FastSAM...";
  
  std::string device = options_.use_gpu ? "cuda" : "cpu";
  std::string model_path = options_.fastsam_model_path;
  if (model_path.empty()) {
      model_path = "FastSAM-x.pt";
  }

  python_bridge_ = std::make_shared<PythonSegmentationBridge>();
  if (python_bridge_->initialize("fast_sam", model_path, device)) {
    is_loaded_ = true;
    LOG(INFO) << "FastSAM initialized successfully via Python bridge";
  } else {
    LOG(ERROR) << "Failed to initialize FastSAM via Python bridge";
    is_loaded_ = false;
  }
}

// Initialize OpenCV methods
void Segmentator::initializeOpenCV() {
  LOG(INFO) << "Initializing OpenCV segmentation methods...";
  is_loaded_ = true;
  LOG(INFO) << "OpenCV segmentation initialized successfully";
}

// Main segmentation function
cv::Mat Segmentator::segment(const cv::Mat& input) {
  if (!is_loaded_) {
    LOG(ERROR) << "Segmentator not loaded!";
    return cv::Mat();
  }
  
  if (input.empty()) {
    LOG(ERROR) << "Input image is empty!";
    return cv::Mat();
  }
  
  cv::Mat result;
  
  switch (options_.model_type) {
    case SegmentationModelType::MobileSAM:
      result = segmentMobileSAM(input);
      break;
    case SegmentationModelType::FastSAM:
      result = segmentFastSAM(input);
      break;
    case SegmentationModelType::OpenCV:
      result = segmentOpenCV(input);
      break;
  }
  
  return result;
}

// MobileSAM segmentation
cv::Mat Segmentator::segmentMobileSAM(const cv::Mat& input) {
  if (python_bridge_) {
      return python_bridge_->segment(input);
  }
  return cv::Mat();
}

// FastSAM segmentation
cv::Mat Segmentator::segmentFastSAM(const cv::Mat& input) {
  if (python_bridge_) {
      return python_bridge_->segment(input);
  }
  return cv::Mat();
}

// OpenCV segmentation
cv::Mat Segmentator::segmentOpenCV(const cv::Mat& input) {
  cv::Mat gray, blurred, thresholded;
  
  // Convert to grayscale
  if (input.channels() == 3) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }
  
  // Apply Gaussian blur
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
  
  // Otsu's thresholding
  cv::threshold(blurred, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
  
  // Apply morphological operations to clean up
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(thresholded, thresholded, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, kernel);
  
  // Apply GrabCut for better segmentation
  cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
  cv::Rect roi(10, 10, input.cols - 20, input.rows - 20);
  
  cv::Mat bgdModel, fgdModel;
  cv::grabCut(input, mask, roi, bgdModel, fgdModel, 
              options_.grabcut_iterations, cv::GC_INIT_WITH_RECT);
  
  // Convert mask to binary
  cv::Mat binary_mask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
  binary_mask.convertTo(binary_mask, CV_8UC1, 255);
  
  return binary_mask;
}

// Segment with prompts (SAM-based)
cv::Mat Segmentator::segmentWithPrompts(const cv::Mat& input, 
                                                  const std::vector<cv::Rect>& bboxes) {
  if (options_.model_type != SegmentationModelType::MobileSAM) {
    LOG(WARNING) << "Prompt-based segmentation only available for MobileSAM";
    return segment(input);
  }
  
  if (bboxes.empty()) {
    return segment(input);
  }
  
  // For MobileSAM with prompts, you would pass bounding boxes to the model
  // This is a simplified version
  cv::Mat combined_mask = cv::Mat::zeros(input.size(), CV_8UC1);
  
  for (const auto& bbox : bboxes) {
    cv::Mat prompt_mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::rectangle(prompt_mask, bbox, cv::Scalar(255), -1);
    
    // In real implementation, you would use this as prompt to MobileSAM
    // For now, just return the union of bounding boxes
    combined_mask = combined_mask | prompt_mask;
  }
  
  return combined_mask;
}

// Instance segmentation (FastSAM only)
std::vector<std::pair<cv::Mat, int>> Segmentator::segmentInstances(const cv::Mat& input) {
  std::vector<std::pair<cv::Mat, int>> instances;
  
  if (options_.model_type != SegmentationModelType::FastSAM) {
    LOG(WARNING) << "Instance segmentation only available for FastSAM";
    return instances;
  }
  
  // Placeholder for FastSAM instance segmentation
  // In real implementation, FastSAM would return multiple masks with class IDs
  
  // Fallback: return single mask with class 0
  cv::Mat mask = segment(input);
  if (!mask.empty()) {
    instances.push_back(std::make_pair(mask, 0));
  }
  
  return instances;
}

// Get class name from ID
std::string Segmentator::getClassName(int class_id) {
  if (python_bridge_) {
      return python_bridge_->getClassName(class_id);
  }
  return "unknown";
}

// Warm up the model
bool Segmentator::warmUp(const cv::Size& image_size) {
  if (!is_loaded_) return false;
  
  cv::Mat dummy = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat result = segment(dummy);
  
  return !result.empty();
}

} // namespace gici