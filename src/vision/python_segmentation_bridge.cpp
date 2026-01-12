#include "gici/vision/python_segmentation_bridge.h"



namespace gici {

class PyGILStateLock {
public:
    PyGILStateLock() {
        state_ = PyGILState_Ensure();
    }
    ~PyGILStateLock() {
        PyGILState_Release(state_);
    }
private:
    PyGILState_STATE state_;
};

PythonSegmentationBridge::PythonSegmentationBridge() 
    : pModule_(nullptr), pClass_(nullptr), pInstance_(nullptr), initialized_(false) {
}

PythonSegmentationBridge::~PythonSegmentationBridge() {
    {
        PyGILStateLock lock;
        if (pInstance_) Py_DECREF(pInstance_);
        if (pClass_) Py_DECREF(pClass_);
        if (pModule_) Py_DECREF(pModule_);
    }
    // Note: Py_Finalize() might be dangerous if other parts of the app use Python
    // or if multiple instances of this class are created/destroyed.
    // For safety in this specific context, we might skip it or guard it.
    // if (initialized_) Py_Finalize(); 
}

bool PythonSegmentationBridge::initialize(const std::string& model_type, 
                                        const std::string& model_path, 
                                        const std::string& device,
                                        const std::map<std::string, double>& config) {
    if (initialized_) return true;

    // Preload Conda's libstdc++ with DEEPBIND to force using its symbols
    // This attempts to expose the GLIBCXX_3.4.30 symbols to the global scope
    const char* conda_libstd = "/home/syl/miniconda3/envs/gici/lib/libstdc++.so.6";
    
    #ifndef RTLD_DEEPBIND
    #define RTLD_DEEPBIND 0x8
    #endif
    
    // We restore LD_LIBRARY_PATH modification because we are now correctly linked against 
    // Conda Python 3.10. Setting LD_LIBRARY_PATH allows all python extensions (cv2, onnxruntime)
    // to find their dependencies (freetype, libffi, etc.) within the Conda environment.
    // This was dangerous when we were on System Python 3.8, but should be safe now.
    const char* conda_lib_dir = "/home/syl/miniconda3/envs/gici/lib";
    std::string new_ld_path = std::string(conda_lib_dir);
    if (const char* old_ld_path = getenv("LD_LIBRARY_PATH")) {
        new_ld_path += ":" + std::string(old_ld_path);
    }
    setenv("LD_LIBRARY_PATH", new_ld_path.c_str(), 1);


    void* handle = dlopen(conda_libstd, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    if (!handle) {
        LOG(WARNING) << "Failed to preload " << conda_libstd << ": " << dlerror();
    } else {
        // LOG(INFO) << "[DEBUG] Successfully preloaded " << conda_libstd << " for ONNX Runtime compatibility." << std::endl;
    }

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    
    // Add script directory to sys.path
    PyObject* sysPath = PySys_GetObject("path");
    
    // Explicitly add conda environment site-packages to sys.path
    // This allows using packages installed in conda (like ultralytics) even when linked against system python
    PyList_Append(sysPath, PyUnicode_FromString("/home/syl/miniconda3/envs/gici/lib/python3.10/site-packages"));
    PyList_Append(sysPath, PyUnicode_FromString("/home/syl/miniconda3/envs/gici/lib/python3.10/lib-dynload"));

    // Explicitly preload the conda libstdc++.so.6 using ctypes to avoid GLIBCXX version issues
    const char* preload_script = 
        "import ctypes\n"
        "import os\n"
        "import sys\n"
        "try:\n"
        "    # Preload libstdc++ (Redundant if C++ did it, but safe)\n"
        "    conda_lib_path = '/home/syl/miniconda3/envs/gici/lib/libstdc++.so.6'\n"
        "    if os.path.exists(conda_lib_path):\n"
        "        ctypes.CDLL(conda_lib_path, mode=ctypes.RTLD_GLOBAL)\n"
        // "        print(f'[DEBUG] Successfully preloaded {conda_lib_path}')\n"
        "    \n"
        // "    print('Python search path:', sys.path)\n"
        "except Exception as e:\n"
        "    print(f'Failed to run preload script: {e}')\n";
    PyRun_SimpleString(preload_script);

    
    // Assuming the script is in /home/syl/GICI-IM/third_party/segmentation
    PyList_Append(sysPath, PyUnicode_FromString("/home/syl/GICI-IM/third_party/segmentation"));
    
    pModule_ = PyImport_ImportModule("segmentation_wrapper");
    if (!pModule_) {
        PyErr_Print();
        LOG(ERROR) << "Failed to load segmentation_wrapper module" << std::endl;
        return false;
    }

    PyObject* pFuncCreate = PyObject_GetAttrString(pModule_, "create_segmentator");
    if (pFuncCreate && PyCallable_Check(pFuncCreate)) {
        pInstance_ = PyObject_CallObject(pFuncCreate, NULL);
        Py_DECREF(pFuncCreate);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        LOG(ERROR) << "Cannot find function create_segmentator" << std::endl;
        return false;
    }

    if (!pInstance_) {
        PyErr_Print();
        LOG(ERROR) << "Failed to create segmentator instance" << std::endl;
        return false;
    }

    // Call initialize method
    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "initialize");
    if (pMethod && PyCallable_Check(pMethod)) {
        // Convert config map to Python dictionary
        PyObject* pConfigDict = PyDict_New();
        for (const auto& pair : config) {
            PyObject* pValue = PyFloat_FromDouble(pair.second);
            PyDict_SetItemString(pConfigDict, pair.first.c_str(), pValue);
            Py_DECREF(pValue);
        }

        // Arguments: model_type, model_path, device, config
        PyObject* pArgs = PyTuple_New(4);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(model_type.c_str()));
        PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(model_path.c_str()));
        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(device.c_str()));
        PyTuple_SetItem(pArgs, 3, pConfigDict); // Steals reference

        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            bool result = PyObject_IsTrue(pValue);
            Py_DECREF(pValue);
            if (!result) {
                LOG(ERROR) << "Python initialize returned false" << std::endl;
                return false;
            }
        } else {
            PyErr_Print();
            LOG(ERROR) << "Call to initialize failed" << std::endl;
            return false;
        }
    } else {
        LOG(ERROR) << "Cannot find initialize method" << std::endl;
        return false;
    }

    initialized_ = true;
    if (PyGILState_Check()) {
        PyEval_SaveThread();
    }
    return true;
}

cv::Mat PythonSegmentationBridge::segment(const cv::Mat& image) {
    if (!initialized_ || !pInstance_) return cv::Mat();

    // Encode image to buffer
    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf);
    
    PyGILStateLock lock;

    // Create bytes object
    PyObject* pBytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buf.data()), buf.size());

    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "segment_from_bytes_return_bytes");
    if (pMethod && PyCallable_Check(pMethod)) {
        PyObject* pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pBytes); // Steals reference

        LOG(INFO) << "Segmentation... ";
        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            LOG(INFO) << "Segmentation success!";
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
                    LOG(ERROR) << "Size mismatch in received mask";
                }
                
                Py_DECREF(pValue);
                return mask;
            }
            Py_DECREF(pValue);
        } else {
            LOG(ERROR) << "Segmentation failed (pValue is NULL). Printing Python error:" << std::endl;
            PyErr_Print();
            LOG(INFO) << "End of Python error." << std::endl;
        }
    } else {
        LOG(ERROR) << "Cannot find segment_from_bytes_return_bytes method" << std::endl;
        Py_DECREF(pBytes);
    }
    
    return cv::Mat();
}

cv::Mat PythonSegmentationBridge::getVisualization(const cv::Mat& image, const cv::Mat& mask) {
    if (!initialized_ || !pInstance_) return image;

    // Encode image to buffer
    std::vector<uchar> img_buf;
    cv::imencode(".jpg", image, img_buf);

    // Encode mask to buffer (raw data)
    size_t mask_size = mask.rows * mask.cols * sizeof(int32_t);
    // Check type
    if (mask.type() != CV_32SC1) {
        LOG(ERROR) << "Mask must be CV_32SC1" << std::endl;
        return image;
    }

    PyGILStateLock lock;

    PyObject* pImgBytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(img_buf.data()), img_buf.size());
    PyObject* pMaskBytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(mask.data), mask_size);

    PyObject* pMethod = PyObject_GetAttrString(pInstance_, "get_visualization_from_bytes");
    if (pMethod && PyCallable_Check(pMethod)) {
        PyObject* pArgs = PyTuple_New(4);
        PyTuple_SetItem(pArgs, 0, pImgBytes); // Steals ref
        PyTuple_SetItem(pArgs, 1, pMaskBytes); // Steals ref
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(mask.rows));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(mask.cols));

        PyObject* pValue = PyObject_CallObject(pMethod, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pMethod);

        if (pValue != NULL) {
            if (PyBytes_Check(pValue)) {
                char* rawData = PyBytes_AsString(pValue);
                Py_ssize_t len = PyBytes_Size(pValue);
                std::vector<uchar> vis_buf(rawData, rawData + len);
                
                cv::Mat vis = cv::imdecode(vis_buf, cv::IMREAD_COLOR);
                Py_DECREF(pValue);
                return vis;
            }
            Py_DECREF(pValue);
        } else {
             PyErr_Print();
        }
    } else {
        LOG(ERROR) << "Cannot find get_visualization_from_bytes" << std::endl;
        Py_DECREF(pImgBytes);
        Py_DECREF(pMaskBytes);
    }
    return image;
}

std::string PythonSegmentationBridge::getClassName(int class_id) {
    if (!initialized_ || !pInstance_) return "unknown";

    PyGILStateLock lock;

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
