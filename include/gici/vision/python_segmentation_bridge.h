#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declaration to avoid including Python.h in the header
typedef struct _object PyObject;

namespace gici {

class PythonSegmentationBridge {
public:
    PythonSegmentationBridge();
    ~PythonSegmentationBridge();

    /**
     * @brief Initialize the Python interpreter and load the model
     * @param model_type "mobile_sam" or "fast_sam"
     * @param model_path Path to the model weights
     * @param device "cpu" or "cuda"
     * @return true if successful
     */
    bool initialize(const std::string& model_type, const std::string& model_path, const std::string& device);

    /**
     * @brief Perform segmentation on an image
     * @param image Input image
     * @return Segmentation mask (CV_8UC1) where pixel values correspond to class IDs or instance IDs
     */
    cv::Mat segment(const cv::Mat& image);

    /**
     * @brief Get the class name for a given class ID
     * @param class_id Class ID
     * @return Class name
     */
    std::string getClassName(int class_id);

private:
    PyObject* pModule_;
    PyObject* pClass_;
    PyObject* pInstance_;
    
    bool initialized_;
};

} // namespace gici
