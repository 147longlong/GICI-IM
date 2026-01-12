/**
* @Function: GNSS/IMU/Camera semi-tightly couple estimator (GNSS solution + IMU raw + camera raw)
*
* @Author  : Cheng Chi
* @Email   : chichengcn@sjtu.edu.cn
*
* Copyright (C) 2023 by Cheng Chi, All rights reserved.
**/
#pragma once

#include "gici/gnss/gnss_loose_estimator_base.h"
#include "gici/imu/imu_estimator_base.h"
#include "gici/vision/visual_estimator_base.h"
#include "gici/fusion/gnss_imu_initializer.h"
#include "gici/integrity/visual_integrity.h"
#include <limits>
#include <memory>

namespace gici {

// GNSS/IMU/Camera SRR couple options
struct GnssImuCameraSrrEstimatorOptions {
  // Frame state window length
  // We only keep GNSS measurements near to keyframes (one-to-one) and throw the others 
  // away after one optimization, because the GNSS measurement errors, especially for 
  // the multipath, are highly correlated between epochs when we have a slow or zero motion.
  // Besides, we need at least 2 GNSS states in window. If current setting cannot ensure 
  // this condition, we will ignore this option and extend the windows length.
  int max_keyframes = 5;

  // GNSS state window length before visual has been initialized
  int max_gnss_window_length_minor = 10;

  // Maximum yaw STD to start visual initialization (deg)
  double min_yaw_std_init_visual = 0.5;

  // Integrity options
  VisualIntegrityOptions integrity_options;
};

// Estimator
class GnssImuCameraSrrEstimator : 
  public GnssLooseEstimatorBase, 
  public VisualEstimatorBase, 
  public ImuEstimatorBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GnssImuCameraSrrEstimator(const GnssImuCameraSrrEstimatorOptions& options, 
               const GnssImuInitializerOptions& init_options, 
               const GnssLooseEstimatorBaseOptions& gnss_loose_base_options, 
               const VisualEstimatorBaseOptions& visual_base_options,
               const ImuEstimatorBaseOptions& imu_base_options,
               const EstimatorBaseOptions& base_options);
  ~GnssImuCameraSrrEstimator();

  // Add measurement
  bool addMeasurement(const EstimatorDataCluster& measurement) override;

  // Estimate current graph
  bool estimate() override;

  // Set initializatin result
  void setInitializationResult(
    const std::shared_ptr<MultisensorInitializerBase>& initializer) override;

protected:
  // Add GNSS measurements and state
  bool addGnssSolutionMeasurementAndState(const GnssSolution& measurement);

  // Add image measurements and state
  bool addImageMeasurementAndState(const FrameBundlePtr& frame_bundle, 
    const SpeedAndBias& speed_and_bias = SpeedAndBias::Zero());

  // Visual initialization
  bool visualInitialization(const FrameBundlePtr& frame_bundle);

  // Marginalization
  bool marginalization(const IdType& type);

  // Marginalization when the new state is a frame state
  bool frameMarginalization();

  // Marginalization when the new state is a GNSS state
  bool gnssMarginalization();

  // Sparsify GNSS states to bound computational load
  void sparsifyGnssStates();

  // Get latest state
  inline State& latestState() override { return states_[latest_state_index_]; }

protected:
  // Options
  GnssImuCameraSrrEstimatorOptions srr_options_;

  // Initialization control
  std::shared_ptr<GnssImuInitializer> gnss_imu_initializer_;
  bool visual_initialized_ = false;
  std::deque<FrameBundlePtr> init_keyframes_;
  std::deque<Solution> init_solution_store_;

  // Status control
  int num_cotinuous_reject_visual_ = 0;

  // Integrity results
  double hpl_ = std::numeric_limits<double>::quiet_NaN();
  double lapl_ = std::numeric_limits<double>::quiet_NaN();
  double lopl_ = std::numeric_limits<double>::quiet_NaN();
  double vpl_ = std::numeric_limits<double>::quiet_NaN();
  double ir_ = std::numeric_limits<double>::quiet_NaN();
  std::unique_ptr<VisualIntegrity> visual_integrity_;

public:
  double getHPL() const { return hpl_; }
  double getLaPL() const { return lapl_; }
  double getLoPL() const { return lopl_; }
  double getVPL() const { return vpl_; }
  double getIR() const { return ir_; }
};

}