#include "gici/vision/segmentator.h"
#include <fstream>

namespace gici {

// Constructor
Segmentator::Segmentator(const SegmentatorOptions& options) 
  : options_(options), is_loaded_(false), fastsam_model_(nullptr) {
  
  // Initialize based on model type
  switch (options_.model_type) {
    case SegmentationModelType::FastSAM:
      initializeFastSAM();
      break;
    case SegmentationModelType::SLIC:
      initializeSLIC();
      break;
  }
}

// Destructor
Segmentator::~Segmentator() {
  // Clean up model resources
  if (fastsam_model_) {
    // Cleanup FastSAM resources
    fastsam_model_ = nullptr;
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
  
  // Pack parameters
  std::map<std::string, double> config;

  python_bridge_ = std::make_shared<PythonSegmentationBridge>();
  if (python_bridge_->initialize("fast_sam", model_path, device, config)) {
    is_loaded_ = true;
    LOG(INFO) << "FastSAM initialized successfully via Python bridge";
    
    // Warm up the model
    LOG(INFO) << "Warming up FastSAM model...";
    if (warmUp()) {
      LOG(INFO) << "FastSAM warmup successful";
    } else {
      LOG(WARNING) << "FastSAM warmup failed";
    }
  } else {
    LOG(ERROR) << "Failed to initialize FastSAM via Python bridge";
    is_loaded_ = false;
  }
}

// Initialize SLIC
void Segmentator::initializeSLIC() {
  LOG(INFO) << "Initializing SLIC segmentation...";
  
  // SLIC runs on CPU usually
  std::string device = "cpu";
  
  // Pack parameters
  std::map<std::string, double> config;

  python_bridge_ = std::make_shared<PythonSegmentationBridge>();
  // SLIC doesn't need a model path
  if (python_bridge_->initialize("slic", "", device, config)) {
    is_loaded_ = true;
    LOG(INFO) << "SLIC initialized successfully via Python bridge";
  } else {
    LOG(ERROR) << "Failed to initialize SLIC via Python bridge";
    is_loaded_ = false;
  }
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
    case SegmentationModelType::FastSAM:
      result = segmentFastSAM(input);
      break;
    case SegmentationModelType::SLIC:
      result = segmentSLIC(input);
      break;
  }
  
  return result;
}

// FastSAM segmentation
cv::Mat Segmentator::segmentFastSAM(const cv::Mat& input) {
  if (python_bridge_) {
      return python_bridge_->segment(input);
  }
  return cv::Mat();
}

// SLIC segmentation
cv::Mat Segmentator::segmentSLIC(const cv::Mat& input) {
  if (python_bridge_) {
      return python_bridge_->segment(input);
  }
  return cv::Mat();
}

cv::Mat Segmentator::getVisualization(const cv::Mat& input, const cv::Mat& mask) {
    if (python_bridge_) {
        return python_bridge_->getVisualization(input, mask);
    }
    return input;
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