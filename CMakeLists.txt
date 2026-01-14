cmake_minimum_required(VERSION 2.8)
project(gici)

# ==========================================
# User Configuration for Python - Paths
# ==========================================
# 1. Conda Environment Path (Used for Python, RPATH, and libstdc++)
set(CONDA_ENV_PATH "/home/syl/miniconda3/envs/gici")

# 2. System Libraries Paths (To avoid conflict with Conda libs)
set(OPENCV_SYSTEM_DIR "/usr/local/lib/cmake/opencv4")
set(YAML_SYSTEM_DIR "/usr/local/lib/cmake/yaml-cpp")

# ==========================================

# Set build flags. 
set(CMAKE_CXX_FLAGS "-std=c++11" )
set(CMAKE_CXX_FLAGS "-fpermissive")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  if (IS_ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
  endif ()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
endif()

# Add third-party libraries
# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)
# Opencv
set(OpenCV_DIR ${OPENCV_SYSTEM_DIR})
find_package(OpenCV 4 REQUIRED)
# Yaml-cpp 
set(yaml-cpp_DIR ${YAML_SYSTEM_DIR})
find_package(yaml-cpp REQUIRED)
# Glog
find_package(glog REQUIRED)
# Ceres-solver
find_package(Ceres REQUIRED)
# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
# Fast feature detector
add_subdirectory(third_party/fast)
# SVO vikit tool-kits
add_subdirectory(third_party/rpg_vikit/vikit_common)
# SVO libraries
add_subdirectory(third_party/rpg_svo)
# RTKLIB
add_definitions(-DENAGLO -DENACMP -DENAGAL -DNFREQ=3 -DNEXOBS=3 -DDLL)
add_subdirectory(third_party/rtklib)
# Find Python
set(Python3_EXECUTABLE "${CONDA_ENV_PATH}/bin/python3")
# Force find_package to look in Conda environment first
set(CMAKE_PREFIX_PATH "${CONDA_ENV_PATH}" ${CMAKE_PREFIX_PATH})
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  include
)

aux_source_directory(src/utility DIR_utility)
aux_source_directory(src/stream DIR_stream)
aux_source_directory(src/gnss DIR_gnss)
aux_source_directory(src/imu DIR_imu)
aux_source_directory(src/vision DIR_vision)
aux_source_directory(src/estimate DIR_estimate)
aux_source_directory(src/fusion DIR_fusion)
aux_source_directory(src/integrity DIR_integrity)
list(APPEND DIR_ALL ${DIR_utility}
                    ${DIR_stream}
                    ${DIR_gnss}
                    ${DIR_imu}
                    ${DIR_vision}
                    ${DIR_estimate}
                    ${DIR_fusion}
                    ${DIR_integrity})

# library
add_library(${PROJECT_NAME} SHARED ${DIR_ALL})
target_compile_definitions(${PROJECT_NAME} PRIVATE CONDA_ENV_PATH="${CONDA_ENV_PATH}")
target_link_libraries(${PROJECT_NAME} 
                      rtklib
                      vikit_common
                      svo
                      glog
                      yaml-cpp
                      ${OpenCV_LIBS}
                      ${Python3_LIBRARIES}
                      ${CONDA_ENV_PATH}/lib/libstdc++.so.6 # Force link against Conda libstdc++
                      )
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# executable
add_executable(gici_main src/gici_main.cpp)
target_link_libraries(gici_main 
                      ${PROJECT_NAME})

# integrity check executable
add_executable(integrity_main src/integrity_main.cpp)
target_link_libraries(integrity_main 
                      ${PROJECT_NAME})