#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
// Legacy Python implementation
#include <Python.h>
#include <vector>
#include <dlfcn.h>
#include "gici/utility/common.h"

// Forward declaration to avoid including Python.h in the header
typedef struct _object PyObject;

namespace gici {

class PythonSegmentationBridge {
public:
    PythonSegmentationBridge();
    ~PythonSegmentationBridge();

    /**
     * @brief Initialize the Python interpreter and load the model
     * @param model_type "mobile_sam" or "fast_sam" or "slic"
     * @param model_path Path to the model weights
     * @param device "cpu" or "cuda"
     * @param config Optional configuration parameters
     * @return true if successful
     */
    bool initialize(const std::string& model_type, 
                   const std::string& model_path, 
                   const std::string& device,
                   const std::map<std::string, double>& config = {});

    /**
     * @brief Perform segmentation on an image
     * @param image Input image
     * @return Segmentation mask (CV_8UC1) where pixel values correspond to class IDs or instance IDs
     */
    cv::Mat segment(const cv::Mat& image);

    /**
     * @brief Get visualization of the segmentation
     * @param image Input image
     * @param mask Segmentation mask
     * @return Visualization image (CV_8UC3)
     */
    cv::Mat getVisualization(const cv::Mat& image, const cv::Mat& mask);

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
