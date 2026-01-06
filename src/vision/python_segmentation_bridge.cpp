#include "gici/vision/python_segmentation_bridge.h"
#include <Python.h>
#include <iostream>
#include <vector>

namespace gici {

PythonSegmentationBridge::PythonSegmentationBridge() 
    : pModule_(nullptr), pClass_(nullptr), pInstance_(nullptr), initialized_(false) {
}

PythonSegmentationBridge::~PythonSegmentationBridge() {
    if (pInstance_) Py_DECREF(pInstance_);
    if (pClass_) Py_DECREF(pClass_);
    if (pModule_) Py_DECREF(pModule_);
    // Note: Py_Finalize() might be dangerous if other parts of the app use Python
    // or if multiple instances of this class are created/destroyed.
    // For safety in this specific context, we might skip it or guard it.
    // if (initialized_) Py_Finalize(); 
}

bool PythonSegmentationBridge::initialize(const std::string& model_type, const std::string& model_path, const std::string& device) {
    if (initialized_) return true;

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    
    // Add script directory to sys.path
    PyObject* sysPath = PySys_GetObject("path");
    // Assuming the script is in /home/dell/sunyulong/GICI-IM/third_party/segmentation
    PyList_Append(sysPath, PyUnicode_FromString("/home/dell/sunyulong/GICI-IM/third_party/segmentation"));
    
    pModule_ = PyImport_ImportModule("segmentation_wrapper");
    if (!pModule_) {
        PyErr_Print();
        std::cerr << "Failed to load segmentation_wrapper module" << std::endl;
        return false;
    }

    PyObject* pFuncCreate = PyObject_GetAttrString(pModule_, "create_segmentator");
    if (pFuncCreate && PyCallable_Check(pFuncCreate)) {
        pInstance_ = PyObject_CallObject(pFuncCreate, NULL);
        Py_DECREF(pFuncCreate);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        std::cerr << "Cannot find function create_segmentator" << std::endl;
        return false;
    }

    if (!pInstance_) {
        PyErr_Print();
        std::cerr << "Failed to create segmentator instance" << std::endl;
        return false;
    }

    // Call initialize method
    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "initialize");
    if (pMethod && PyCallable_Check(pMethod)) {
        PyObject* pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(model_type.c_str()));
        PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(model_path.c_str()));
        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(device.c_str()));

        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            bool result = PyObject_IsTrue(pValue);
            Py_DECREF(pValue);
            if (!result) {
                std::cerr << "Python initialize returned false" << std::endl;
                return false;
            }
        } else {
            PyErr_Print();
            std::cerr << "Call to initialize failed" << std::endl;
            return false;
        }
    } else {
        std::cerr << "Cannot find initialize method" << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

cv::Mat PythonSegmentationBridge::segment(const cv::Mat& image) {
    if (!initialized_ || !pInstance_) return cv::Mat();

    // Encode image to buffer
    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf);
    
    // Create bytes object
    PyObject* pBytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buf.data()), buf.size());

    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "segment_from_bytes_return_bytes");
    if (pMethod && PyCallable_Check(pMethod)) {
        PyObject* pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pBytes); // Steals reference

        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            // Expecting tuple (bytes, rows, cols)
            if (PyTuple_Check(pValue) && PyTuple_Size(pValue) == 3) {
                PyObject* pMaskBytes = PyTuple_GetItem(pValue, 0);
                PyObject* pRows = PyTuple_GetItem(pValue, 1);
                PyObject* pCols = PyTuple_GetItem(pValue, 2);

                char* rawData = PyBytes_AsString(pMaskBytes);
                Py_ssize_t len = PyBytes_Size(pMaskBytes);
                int rows = (int)PyLong_AsLong(pRows);
                int cols = (int)PyLong_AsLong(pCols);

                cv::Mat mask(rows, cols, CV_32SC1);
                if (len == rows * cols * sizeof(int32_t)) {
                    memcpy(mask.data, rawData, len);
                } else {
                    std::cerr << "Size mismatch in received mask" << std::endl;
                }
                
                Py_DECREF(pValue);
                return mask;
            }
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        std::cerr << "Cannot find segment_from_bytes_return_bytes method" << std::endl;
        Py_DECREF(pBytes);
    }
    
    return cv::Mat();
}

std::string PythonSegmentationBridge::getClassName(int class_id) {
    if (!initialized_ || !pInstance_) return "unknown";

    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "get_class_name");
    if (pMethod && PyCallable_Check(pMethod)) {
        PyObject* pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(class_id));

        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            std::string result = "";
            PyObject* pStr = PyObject_Str(pValue);
            if (pStr) {
                result = PyUnicode_AsUTF8(pStr);
                Py_DECREF(pStr);
            }
            Py_DECREF(pValue);
            return result;
        } else {
            PyErr_Print();
        }
    }
    return "unknown";
}

} // namespace gici
