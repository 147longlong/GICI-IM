cmake_minimum_required(VERSION 2.8)
project(IM)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: ${CMAKE_BUILD_TYPE} " )

find_package(Eigen3 QUIET)
find_package(Boost QUIET)
if(NOT EIGEN3_FOUND)
    set(EIGEN3_INCLUDE_DIRS "D:/Code/FAIM-integrity/third_party/eigen-3.4.0")
endif()
if(NOT Boost_FOUND)
    set(Boost_INCLUDE_DIRS "D:/Code/FAIM-integrity/third_party/boost_1_84_0" )
endif()
message( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
message( STATUS "Eigen3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}.")


include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/ DIR_SRC)
add_library(INTEGRITY_LIB SHARED ${DIR_SRC})

add_executable(main main.cpp)
add_executable(test test.cpp)
target_link_libraries(main INTEGRITY_LIB)
target_link_libraries(test INTEGRITY_LIB)