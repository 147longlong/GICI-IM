/**
* @Function: Visual Integrity Monitoring using MHSS
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#include "gici/integrity/visual_integrity.h"
#include <algorithm>
#include <cctype>
#include <atomic>
#include <boost/math/distributions/chi_squared.hpp>

#include "gici/gnss/doppler_error.h"
#include "gici/gnss/phaserange_error_dd.h"
#include "gici/gnss/pseudorange_error_dd.h"
#include "gici/imu/imu_error.h"

namespace gici {

namespace {
double percentileValue(std::vector<double> values, const double percentile)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();

    const double p = std::min(100.0, std::max(0.0, percentile));
    const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
    const int lo = static_cast<int>(std::floor(rank));
    const int hi = static_cast<int>(std::ceil(rank));

    std::nth_element(values.begin(), values.begin() + lo, values.end());
    const double vlo = values[lo];
    if (lo == hi) return vlo;

    std::nth_element(values.begin(), values.begin() + hi, values.end());
    const double vhi = values[hi];
    const double w = rank - static_cast<double>(lo);
    return vlo + (vhi - vlo) * w;
}

void scaleSigmaSpikesInColumn(Eigen::MatrixXd& matrix,
                              const int col,
                              const double percentile,
                              const double max_excess)
{
    if (matrix.rows() == 0 || col < 0 || col >= matrix.cols() || max_excess <= 0.0) return;

    std::vector<double> finite_values;
    finite_values.reserve(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        const double v = matrix(i, col);
        if (std::isfinite(v)) finite_values.push_back(v);
    }

    const double threshold = percentileValue(std::move(finite_values), percentile);
    if (!std::isfinite(threshold)) return;

    for (int i = 0; i < matrix.rows(); ++i) {
        const double v = matrix(i, col);
        if (!std::isfinite(v) || v <= threshold) continue;
        const double excess = v - threshold;
        matrix(i, col) = threshold + max_excess * std::tanh(excess / max_excess);
    }
}

const GnssMeasurement* selectGnssMeasurementForTimestamp(
    const double timestamp,
    const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs)
{
    if (gnss_measurement_pairs == nullptr || gnss_measurement_pairs->empty() || !std::isfinite(timestamp)) {
        LOG(WARNING) << "No GNSS measurement pairs available or invalid timestamp: " << std::setprecision(12) << std::fixed << timestamp;
        return nullptr;
    }

    const GnssMeasurement* best_measurement = nullptr;
    double min_dt = std::numeric_limits<double>::infinity();
    for (const auto& measurement_pair : *gnss_measurement_pairs) {
        const double dt = std::fabs(measurement_pair.first.timestamp - timestamp);
        if (dt < min_dt) {
            min_dt = dt;
            best_measurement = &measurement_pair.first;
        }
    }

    if (min_dt > 0.05) {
        // LOG(WARNING) << "No closely matched GNSS measurement found for timestamp: " << std::setprecision(12) << std::fixed << timestamp
        //              << ", closest GNSS measurement timestamp: " << std::setprecision(12) << std::fixed << (best_measurement ? best_measurement->timestamp : 0.0)
        //              << ", min_dt: " << std::setprecision(3) << std::fixed << min_dt;
    }

    return best_measurement;
}
}  // namespace


VisualIntegrity::VisualIntegrity(const VisualIntegrityOptions& options)
    : options_(options), LaPL_(std::numeric_limits<double>::quiet_NaN()), LoPL_(std::numeric_limits<double>::quiet_NaN()), VPL_(std::numeric_limits<double>::quiet_NaN()), IR_(std::numeric_limits<double>::quiet_NaN())
{   
    is_first_ = true;
}


VisualIntegrity::~VisualIntegrity()
{
}


// The monitor function for real-time integrity monitoring
bool VisualIntegrity::monitor(const std::deque<State>& states, size_t state_index, const Graph* graph,
                    const FramePtr& frame, const PointMap& landmarks_map,
                    const GnssMeasurement* measurement_rov,
                    const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs)
{
    State state = states[state_index];
    timestamp_ = state.timestamp;
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return false;

    Eigen::MatrixXd J_all;
    Eigen::VectorXd r_all;
    Eigen::MatrixXd sig2_int;
    Eigen::MatrixXd sig2_acc;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows;
    std::map<uint64_t, std::vector<int>> lm_to_J_rows;
    std::map<uint64_t, int> curr_lm_to_object_ids;
    std::map<uint64_t, int> lm_to_object_ids;
    std::map<std::string, std::vector<int>> curr_sat_to_J_rows;
    std::map<std::string, std::vector<int>> sat_to_J_rows;
    std::map<uint64_t, std::vector<int>> curr_imu_to_J_rows;
    std::map<uint64_t, std::vector<int>> imu_to_J_rows;
    std::vector<int> curr_pose_J_cols;

    if (!prepareLinearSystem(frame, states, state_index, graph, landmarks_map, gnss_measurement_pairs,
                             J_all, r_all, sig2_int, sig2_acc,
                             curr_lm_to_J_rows, lm_to_J_rows, curr_lm_to_object_ids, lm_to_object_ids,
                             curr_sat_to_J_rows,
                             sat_to_J_rows,
                             curr_imu_to_J_rows,
                             imu_to_J_rows,
                             curr_pose_J_cols)) {
        return false;
    }
    

    computeIntegrityMetrics(J_all, r_all, sig2_int, sig2_acc, 
                             curr_lm_to_J_rows, curr_lm_to_object_ids, 
                             curr_sat_to_J_rows,
                             curr_imu_to_J_rows,
                             curr_pose_J_cols);

    // Log results
    LOG(INFO) << std::scientific << std::setprecision(4)
              << "timestamp: " << timestamp_
              << ", LaPL: " << LaPL_ << " m"
              << ", LoPL: " << LoPL_ << " m"
              << ", VPL: " << VPL_ << " m";

    return (LaPL_ < options_.LaAL && LoPL_ < options_.LoAL && VPL_ < options_.VAL);
}

// Function to save integrity input information for post-processing
void VisualIntegrity::saveSnapshot(const std::deque<State>& states, size_t state_index, const Graph* graph,
                    const FramePtr& frame, const PointMap& landmarks_map,
                    const GnssMeasurement* measurement_rov,
                    const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs)
{
    if (is_first_) {
        std::ofstream ofs(options_.snapshot_file, std::ios::binary | std::ios::trunc);
        if (ofs.is_open()) {
            serializeOptions(options_, ofs);
            ofs.close();
            LOG(INFO) << "Created snapshot file: " << options_.snapshot_file;
            LOG(INFO) << "Serialized options to snapshot file.";
            is_first_ = false;
        }
    }

    const State& state = states[state_index];
    if (!const_cast<State&>(state).valid()) return;

    CHECK(state.id.type() == IdType::cPose || state.id.type() == IdType::gPose)
        << "State is not a pose type, state id type = " << static_cast<int>(state.id.type());

    timestamp_ = state.timestamp;
    if (last_timestamp_ > 0 && (timestamp_ - last_timestamp_) < 1 / options_.snapshot_freq) {
        LOG(INFO) << "The save snapshot frequency: " << options_.snapshot_freq << ", skipped timestamp: " << std::setprecision(6) << std::fixed << timestamp_;
        return;
    }

    // if (state.id.type() == IdType::gPose && consecutive_gpose_saved_ >= 3) {
    //     LOG(INFO) << "Skipped gPose snapshot at timestamp: " << std::setprecision(6) << std::fixed << timestamp_
    //               << ", waiting for cPose after " << consecutive_gpose_saved_ << " consecutive gPose snapshots.";
    //     return;
    // }

    // last_timestamp_ = timestamp_;
    // if (state.id.type() == IdType::gPose) {
    //     ++consecutive_gpose_saved_;
    // } else {
    //     consecutive_gpose_saved_ = 0;
    // }

    IntegritySnapshot snapshot;
    snapshot.timestamp = state.timestamp;

    if (!prepareLinearSystem(frame, states, state_index, graph, landmarks_map, gnss_measurement_pairs,
                             snapshot.J_all, snapshot.r_all, snapshot.sig2_int, snapshot.sig2_acc, 
                             snapshot.curr_lm_to_J_rows, snapshot.lm_to_J_rows, snapshot.curr_lm_to_object_ids, snapshot.lm_to_object_ids,
                             snapshot.curr_sat_to_J_rows,
                             snapshot.sat_to_J_rows,
                             snapshot.curr_imu_to_J_rows,
                             snapshot.imu_to_J_rows,
                             snapshot.curr_pose_J_cols)) {
        return;
    }

    if (!options_.snapshot_file.empty()) {
        std::ofstream ofs(options_.snapshot_file, std::ios::binary | std::ios::app);
        if (ofs.is_open()) {
            serializeSnapshot(snapshot, ofs);
            ofs.close();
            LOG(INFO) << std::fixed << std::setprecision(6) << "Snapshot serialized to " << options_.snapshot_file << " for timestamp " << snapshot.timestamp;
        } else {
            LOG(ERROR) << "Failed to open snapshot file: " << options_.snapshot_file;
        }
    } else {
        LOG(WARNING) << "Snapshot file not set, skipping save.";
    }
}


void VisualIntegrity::processSnapshotsFromFile(const std::string& filename)
{
    LOG(INFO) << "Processing snapshots from file: " << filename;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        LOG(ERROR) << "Failed to open snapshot file: " << filename;
        return;
    }
    VisualIntegrityOptions snapfile_opts;
    if (is_first_ && ifs.peek() != EOF && !options_.yaml_options) {
        deserializeOptions(options_, ifs);
        LOG(INFO) << "Read options from snapshot file: " << filename;
        is_first_ = false;
    } else{
        deserializeOptions(snapfile_opts, ifs);
        is_first_ = false;
    }

    std::ofstream csv_out;
    if (!csv_output_file_.empty()) {
        csv_out.open(csv_output_file_, std::ios::trunc);
        csv_out << "Timestamp,HPL,VPL,LaPL,LoPL,IR" << std::endl;
    }

    struct IntegrityResult {
        double sod; // Seconds of day
        double timestamp;
        double hpl, vpl, xpl, ypl, ir;
    };
    std::vector<IntegrityResult> results_list;

    double last_processed_timestamp = -1.0;
    int processed_count = 0;
    int last_progress = -1;
    
    while (ifs.peek() != EOF) {
        IntegritySnapshot snapshot;
        deserializeSnapshot(snapshot, ifs);
        if (ifs.fail()) break;
        timestamp_ = snapshot.timestamp;

        if (last_processed_timestamp > 0 && (timestamp_ - last_processed_timestamp) < 1.0/options_.snapshot_freq) {
            continue;
        }
        last_processed_timestamp = timestamp_;
                    
        if (timestamp_ < options_.start_timestamp) {
            LOG(WARNING) << std::fixed << std::setprecision(6)  << "Skipped Timestamp: " << timestamp_;
            continue;
        }
        if (options_.end_timestamp > 0.0 && timestamp_ > options_.end_timestamp) {
             LOG(WARNING) << std::fixed << std::setprecision(6)  << "Skipped Timestamp (end): " << timestamp_;
             break;
        }

        VPL_ = std::numeric_limits<double>::quiet_NaN();
        LaPL_ = std::numeric_limits<double>::quiet_NaN();
        LoPL_ = std::numeric_limits<double>::quiet_NaN();
        HPL_ = std::numeric_limits<double>::quiet_NaN();
        IR_ = 0;

        LOG(INFO) << std::fixed << std::setprecision(6) << "Timestamp: " << timestamp_;

        if (options_.yaml_options && (options_.use_complex_gnss_cov != snapfile_opts.use_complex_gnss_cov 
                                    || options_.use_complex_imu_cov != snapfile_opts.use_complex_imu_cov 
                                    || options_.use_complex_visual_cov != snapfile_opts.use_complex_visual_cov)) {
            std::vector<int> gnss_rows;
            for (const auto& kv : snapshot.sat_to_J_rows) {
                gnss_rows.insert(gnss_rows.end(), kv.second.begin(), kv.second.end());
            }
            std::vector<int> imu_rows;
            for (const auto& kv : snapshot.imu_to_J_rows) {
                imu_rows.insert(imu_rows.end(), kv.second.begin(), kv.second.end());
            }
            std::vector<int> visual_rows;
            for (const auto& kv : snapshot.lm_to_J_rows) {
                visual_rows.insert(visual_rows.end(), kv.second.begin(), kv.second.end());
            }
            std::vector<int> others_rows;
            for (int i = 0; i < snapshot.r_all.size(); ++i) {
                if (std::find(gnss_rows.begin(), gnss_rows.end(), i) == gnss_rows.end() &&
                    std::find(imu_rows.begin(), imu_rows.end(), i) == imu_rows.end() &&
                    std::find(visual_rows.begin(), visual_rows.end(), i) == visual_rows.end()) {
                    others_rows.push_back(i);
                }
            }
            if (!options_.use_complex_gnss_cov && options_.use_complex_gnss_cov != snapfile_opts.use_complex_gnss_cov && !gnss_rows.empty()) {
                LOG(INFO) << "Updating GNSS covariance complexity for post-processing analysis.";
                for (int i = 0; i < snapshot.r_all.size(); ++i) {
                    if (std::find(gnss_rows.begin(), gnss_rows.end(), i) != gnss_rows.end()) {
                        snapshot.sig2_int(i, i) = std::pow(options_.simple_gnss_sigma, 2);
                        snapshot.sig2_acc(i, i) = std::pow(options_.simple_gnss_sigma, 2);
                        for (int j = 0; j < snapshot.r_all.size(); ++j) {
                            if (j != i) {
                                snapshot.sig2_int(i, j) = 0.0;
                                snapshot.sig2_acc(i, j) = 0.0;
                            }
                        }
                    }
                }
            }
            if (!options_.use_complex_imu_cov && options_.use_complex_imu_cov != snapfile_opts.use_complex_imu_cov && !imu_rows.empty()) {
                LOG(INFO) << "Updating IMU covariance complexity for post-processing analysis.";
                for (int i = 0; i < snapshot.r_all.size(); ++i) {
                    if (std::find(imu_rows.begin(), imu_rows.end(), i) != imu_rows.end()) {
                        snapshot.sig2_int(i, i) = std::pow(options_.simple_imu_sigma, 2);
                        snapshot.sig2_acc(i, i) = std::pow(options_.simple_imu_sigma, 2);
                        for (int j = 0; j < snapshot.r_all.size(); ++j) {
                            if (j != i) {
                                snapshot.sig2_int(i, j) = 0.0;
                                snapshot.sig2_acc(i, j) = 0.0;
                            }
                        }
                    }
                }
            }
            if (!options_.use_complex_visual_cov && options_.use_complex_visual_cov != snapfile_opts.use_complex_visual_cov && !visual_rows.empty()) {
                LOG(INFO) << "Updating visual covariance complexity for post-processing analysis.";
                for (int i = 0; i < snapshot.r_all.size(); ++i) {
                    if (std::find(visual_rows.begin(), visual_rows.end(), i) != visual_rows.end()) {
                        snapshot.sig2_int(i, i) = std::pow(options_.simple_visual_sigma, 2);
                        snapshot.sig2_acc(i, i) = std::pow(options_.simple_visual_sigma, 2);
                        for (int j = 0; j < snapshot.r_all.size(); ++j) {
                            if (j != i) {
                                snapshot.sig2_int(i, j) = 0.0;
                                snapshot.sig2_acc(i, j) = 0.0;
                            }
                        }
                    }
                }
            }
             if (!options_.use_complex_others_cov && options_.use_complex_others_cov != snapfile_opts.use_complex_others_cov && !others_rows.empty()) {
                LOG(INFO) << "Updating others covariance complexity for post-processing analysis.";
                for (int i = 0; i < snapshot.r_all.size(); ++i) {
                    if (std::find(others_rows.begin(), others_rows.end(), i) != others_rows.end()) {
                        snapshot.sig2_int(i, i) = std::pow(options_.simple_others_sigma, 2);
                        snapshot.sig2_acc(i, i) = std::pow(options_.simple_others_sigma, 2);
                        for (int j = 0; j < snapshot.r_all.size(); ++j) {
                            if (j != i) {
                                snapshot.sig2_int(i, j) = 0.0;
                                snapshot.sig2_acc(i, j) = 0.0;
                            }
                        }
                    }
                }
             }
        }

        // You can select use "curr_*_to_J_rows" or the full "*_to_J_rows" based on your needs. Here we use curr_*_to_J_rows for a more real-time monitoring perspective.
        computeIntegrityMetrics(snapshot.J_all, snapshot.r_all, snapshot.sig2_int, snapshot.sig2_acc, 
                                snapshot.curr_lm_to_J_rows, snapshot.curr_lm_to_object_ids, 
                                snapshot.curr_sat_to_J_rows,
                                snapshot.curr_imu_to_J_rows,
                                snapshot.curr_pose_J_cols);

        // Debug input
        #if 0
        
        saveEigenMatrixToFile(snapshot.J_all, debug_dir + "J_all_output" + std::to_string(timestamp_)  + ".txt");
        // saveEigenMatrixToFile(snapshot.r_all, debug_dir + "r_all_output" + std::to_string(timestamp_)  + ".txt");
        saveEigenMatrixToFile(snapshot.sig2_int, debug_dir + "sig2_int_output" + std::to_string(timestamp_)  + ".txt");
        // saveEigenMatrixToFile(snapshot.sig2_acc, debug_dir + "sig2_acc_output" + std::to_string(timestamp_)  + ".txt");
        saveMeasDebugFile(debug_dir + "landmark_curr_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.curr_lm_to_J_rows, "landmark_id", "landmark jacobian shape", &snapshot.curr_lm_to_object_ids);
        saveMeasDebugFile(debug_dir + "landmark_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.lm_to_J_rows, "landmark_id", "landmark jacobian shape", &snapshot.lm_to_object_ids);
        saveMeasDebugFile(debug_dir + "gnss_sat_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.sat_to_J_rows, "PRN", "GNSS sat jacobian shape");
        saveMeasDebugFile(debug_dir + "gnss_curr_sat_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.curr_sat_to_J_rows, "PRN", "GNSS sat jacobian shape");
        saveMeasDebugFile(debug_dir + "imu_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.imu_to_J_rows, "imu_residual_id", "IMU jacobian shape");
        saveMeasDebugFile(debug_dir + "imu_curr_jacobian_shape_" + std::to_string(timestamp_) + ".txt",
            timestamp_, snapshot.curr_imu_to_J_rows, "imu_residual_id", "IMU jacobian shape");
        #endif


        #if 0
        int num_residual = snapshot.r_all.size();
        int num_state_vars = snapshot.J_all.cols();
        int num_meas = snapshot.curr_lm_to_J_rows.size();

        // Group measurements by object ID
        std::map<int, int> object_counts;
        int independent_faults = 0;

        for (const auto& kv : snapshot.curr_lm_to_J_rows) {
            uint64_t lm_id = kv.first;
            int object_id = -1;
            if (snapshot.curr_lm_to_object_ids.count(lm_id)) {
                object_id = snapshot.curr_lm_to_object_ids.at(lm_id);
            }
            
            // Assuming object_id >= 0 indicates a valid group/object
            if (object_id >= 0) {
                object_counts[object_id]++;
            } else {
                independent_faults++;
            }
        }
        
        std::vector<double> p_prior_groups;
        p_prior_groups.reserve(object_counts.size() + independent_faults);
        LOG(INFO) << "Number of object groups with multiple measurements: " << object_counts.size();
        LOG(INFO) << "Number of independent measurements: " << independent_faults;

        // Add probabilities for object groups: P = 1 - (1 - p)^n
        for (const auto& pair : object_counts) {
            int n_ms = pair.second;
            double p_group = 1.0 - std::pow(1.0 - options_.visual_prior_fault_probability, n_ms);
            p_prior_groups.push_back(p_group);
        }
        // Add probabilities for independent faults
        for (int i = 0; i < independent_faults; ++i) {
            p_prior_groups.push_back(options_.visual_prior_fault_probability);
        }

        int num_groups = p_prior_groups.size();
        int N_fault_max = determineNfaultmax(p_prior_groups, options_.P_THRES);
        LOG(INFO) << "The maximum simultanous faults need to monitor = " << N_fault_max << ", in P_THRES = " << options_.P_THRES;
        
        long long subsetsize = 0;
        for(int j = 0; j <= N_fault_max; ++j){
            subsetsize = subsetsize + nchoosek(num_groups, j);
        }
        LOG(INFO) << "The total subset size = " << subsetsize << " (Groups: " << num_groups << ", Meas: " << num_meas << ")";

        std::ofstream debug_file("/home/syl/GICI-IM/results/subset_info_super.txt", std::ios::app);
        if (debug_file.is_open()) {
            //# time, num_meas, num_groups, N_fault_max, subsetsize, num_residual, num_state_vars
            debug_file << std::fixed << std::setprecision(6) << timestamp_ << " " 
                       << num_meas << " " 
                    //    << num_groups << " "
                       << N_fault_max << " " 
                       << subsetsize << " "
                       << num_residual << " " 
                       << num_state_vars << std::endl;
            debug_file.close();
        }
        #endif

        #if 0
        int num_residual = snapshot.r_all.size();
        int num_state_vars = snapshot.J_all.cols();
        int num_meas = snapshot.curr_lm_to_J_rows.size();

        std::vector<double> p_prior(num_meas, options_.visual_prior_fault_probability);
        int N_fault_max = determineNfaultmax(p_prior, options_.P_THRES);
        LOG(INFO) << "The maximum simultanous faults need to monitor = " << N_fault_max << ", in P_THRES = " << options_.P_THRES;
        int subsetsize = 0;
        for(int j = 0; j <= N_fault_max;++j){
            subsetsize = subsetsize + nchoosek((num_meas),j);
        }
        LOG(INFO) << "The total subset size = " << subsetsize;

        std::ofstream debug_file("/home/dell/sunyulong/GICI-IM/results/subset_info_1e_12.txt", std::ios::app);
        if (debug_file.is_open()) {
            //# time, num_meas, N_fault_max, subsetsize, num_residual, num_state_vars
            debug_file << std::fixed << std::setprecision(6) << timestamp_ << " " 
                       << num_meas << " " 
                       << N_fault_max << " " 
                       << subsetsize << " "
                       << num_residual << " " 
                       << num_state_vars << std::endl;
            debug_file.close();
        }
        #endif
        LOG(INFO) << std::fixed << std::setprecision(6) 
                  << "Timestamp: " << timestamp_
                  << ", LaPL: " << LaPL_ << " m"
                  << ", LoPL: " << LoPL_ << " m"
                  << ", VPL: " << VPL_ << " m";

        if (csv_out.is_open()) {
            csv_out << std::fixed << std::setprecision(6) << timestamp_ << "," 
                << HPL_ << "," << VPL_ << "," << LaPL_ << "," << LoPL_ << "," << IR_ << std::endl;
        }

        if (LaPL_ > 1e4 || LoPL_ > 1e4 || VPL_ > 1e4 || std::isnan(VPL_) || std::isnan(LaPL_) || std::isnan(LoPL_)) {
            LOG(WARNING) << "Abnormally large VPL detected, at timestamp: "<< std::fixed << std::setprecision(6) << timestamp_;
        }

        // Free memory for the processed snapshot to avoid OOM
        snapshot.J_all.resize(0, 0);
        snapshot.r_all.resize(0);
        snapshot.sig2_int.resize(0, 0);
        snapshot.sig2_acc.resize(0, 0);
        snapshot.curr_lm_to_J_rows.clear();
        snapshot.lm_to_J_rows.clear();
        snapshot.curr_sat_to_J_rows.clear();
        snapshot.sat_to_J_rows.clear();
        snapshot.curr_imu_to_J_rows.clear();
        snapshot.imu_to_J_rows.clear();
        snapshot.curr_lm_to_object_ids.clear();
        snapshot.lm_to_object_ids.clear();
        snapshot.curr_pose_J_cols.clear();

        // Generate timestamp for NMEA matching
        gtime_t t = gici::gnss_common::doubleToGtime(timestamp_);
        t = utc2gpst(t);
        t = gpst2utc(t);
        double ep[6];
        time2epoch(t, ep);
        
        double sod = ep[3] * 3600.0 + ep[4] * 60.0 + ep[5];
        results_list.push_back({sod, snapshot.timestamp, HPL_, VPL_, LaPL_, LoPL_, IR_});
        LOG(INFO) << "===========================================================================";
    }

    if (csv_out.is_open()) csv_out.close();

    // Update NMEA file
    if (output_file_.empty()) return;
    
    LOG(INFO) << "Updating NMEA file: " << output_file_;
    std::ifstream in(output_file_);
    if (!in.is_open()) {
        LOG(ERROR) << "Cannot open output file for reading: " << output_file_;
        return;
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    in.close();

    std::ofstream out(output_file_, std::ios::trunc);
    if (!out.is_open()) {
        LOG(ERROR) << "Cannot open output file for writing: " << output_file_;
        return;
    }

    const double kImMatchToleranceSec = 0.1;
    std::vector<bool> result_used(results_list.size(), false);

    for (auto& l : lines) {
        // Check if line is $..IM and contains timestamp
        if (l.size() > 6 && l.substr(3, 3) == "IM,") {
            // Extract timestamp
            size_t first_comma = l.find(',');
            size_t second_comma = l.find(',', first_comma + 1);
            if (first_comma != std::string::npos && second_comma != std::string::npos) {
                std::string ts_str = l.substr(first_comma + 1, second_comma - first_comma - 1);
                
                if (ts_str.size() >= 6) {
                    double h = std::stod(ts_str.substr(0, 2));
                    double m = std::stod(ts_str.substr(2, 2));
                    double s = std::stod(ts_str.substr(4));
                    double nmea_sod = h * 3600.0 + m * 60.0 + s;

                    int best_idx = -1;
                    double best_dt = kImMatchToleranceSec;
                    for (size_t i = 0; i < results_list.size(); ++i) {
                        if (result_used[i]) continue;
                        const double dt = std::abs(results_list[i].sod - nmea_sod);
                        if (dt < best_dt) {
                            best_dt = dt;
                            best_idx = static_cast<int>(i);
                        }
                    }

                    if (best_idx >= 0) {
                        const auto& item = results_list[best_idx];
                        // Found nearest unmatched result within tolerance, replace this line.
                        std::string talker = l.substr(1, 2);
                        char buf[256];
                        char* p = buf;
                        p += sprintf(p, "$%sIM,%s,%.4e,%.4e,%.4e,%.4e",
                            talker.c_str(), ts_str.c_str(), item.xpl, item.ypl, item.vpl, item.ir);

                        // Calculate checksum
                        char sum = 0;
                        for (char* q = buf + 1; *q; q++) sum ^= *q;
                        sprintf(p, "*%02X", sum);

                        l = std::string(buf);
                        result_used[best_idx] = true;
                        LOG(INFO) << "Updated NMEA line for timestamp " << ts_str
                                    << " (matched dt=" << std::fixed << std::setprecision(3) << best_dt << "s)";
                    }
                }
            }
        }
        out << l << "\n"; 
    }
    out.close();
    
    LOG(INFO) << "Finished processing snapshots and updating file.";
}

namespace {
    struct BalancePoint {
        double x, y;
        int id;
    };

    void balanceObjectIds(std::vector<BalancePoint>& points) {
        if (points.empty()) return;

        const double MAX_MERGE_DIST_SQ = 100.0 * 100.0;
        const int SMALL_CLUSTER_THRESHOLD = 2;
        const int LARGE_CLUSTER_THRESHOLD = 4;
        const int MIN_TOTAL_IDS_THRESHOLD = 20;
        const int MIN_REQUIRED_IDS = 10;
        const int MAX_PASSES = 6;

        // 1. Handling Large Clusters
        // Use a multi-pass approach to stabilize offloading
        bool unstable = true;
        int pass = 0;
        
        while(unstable && pass < MAX_PASSES) {
            unstable = false;
            pass++;
            
            // Re-build cluster map
            std::map<int, std::vector<size_t>> clusters;
            int max_id = 0;
            for (const auto& p : points) if (p.id > max_id) max_id = p.id;
            
            for (size_t i = 0; i < points.size(); ++i) {
                if (points[i].id >= 0) clusters[points[i].id].push_back(i);
            }

            // Iterate over clusters
            // Note: modifying points[i].id invalidates subsequent lookups provided we iterate safely
            for (auto& pair : clusters) {
                int pid = pair.first;
                std::vector<size_t>& members = pair.second;
                
                if (members.size() > LARGE_CLUSTER_THRESHOLD) {
                    
                    // Strategy 1: Offload to nearby small clusters
                    // Find a candidate point to move (one that is closest to a small cluster)
                    int best_mem_idx = -1;
                    int target_id = -1;
                    double best_dist = MAX_MERGE_DIST_SQ;

                    for(size_t idx : members) {
                        for(size_t j = 0; j < points.size(); ++j) {
                            if (points[j].id == pid || points[j].id < 0) continue;
                            
                            // Check target cluster size (must be small)
                            // We need real-time size, but approximation from map is okay for this pass
                            size_t target_size = 0;
                            if (clusters.count(points[j].id)) target_size = clusters[points[j].id].size();
                            
                            if (target_size < SMALL_CLUSTER_THRESHOLD) {
                                double dist = (points[idx].x - points[j].x)*(points[idx].x - points[j].x) + 
                                              (points[idx].y - points[j].y)*(points[idx].y - points[j].y);
                                if (dist < best_dist) {
                                    best_dist = dist;
                                    best_mem_idx = idx;
                                    target_id = points[j].id;
                                }
                            }
                        }
                    }

                    if (best_mem_idx != -1) {
                         // Found a move
                         points[best_mem_idx].id = target_id;
                         unstable = true;
                         // We modified the state, simple break to re-evaluate or continue to next cluster?
                         // Let's continue to next cluster to avoid iterator issues with 'clusters' loop if we were more aggressive
                         continue; 
                    }
                    
                    // Strategy 2: Split in two (New ID)
                    // No nearby small cluster found, so we must split.
                    
                    // Calculate bounds to find largest spread axis
                    double min_x = std::numeric_limits<double>::max();
                    double max_x = std::numeric_limits<double>::lowest();
                    double min_y = std::numeric_limits<double>::max();
                    double max_y = std::numeric_limits<double>::lowest();

                    for(size_t idx : members) { 
                        if(points[idx].x < min_x) min_x = points[idx].x;
                        if(points[idx].x > max_x) max_x = points[idx].x;
                        if(points[idx].y < min_y) min_y = points[idx].y;
                        if(points[idx].y > max_y) max_y = points[idx].y;
                    }

                    bool split_by_x = (max_x - min_x) > (max_y - min_y);
                    
                    // Sort members by the axis with largest spread
                    std::sort(members.begin(), members.end(), [&](size_t a, size_t b) {
                        if (split_by_x) return points[a].x < points[b].x;
                        return points[a].y < points[b].y;
                    });
                    
                    // Assign half to NEW ID (spatial split)
                    int new_id = ++max_id;
                    size_t split_start_idx = members.size() / 2;
                    
                    for(size_t k = split_start_idx; k < members.size(); ++k) {
                        points[members[k]].id = new_id;
                    }
                    
                    unstable = true; 
                    // Break outer loop to rebuild map? Or just continue? 
                    // Rebuilding is safer because we introduced a new ID
                    // But we can just continue to next cluster in the map
                }
            }
        }

        // 2. Handling Small Clusters & Ensures Minimum ID Count
        std::map<int, int> counts;
        int max_id_final = -1;
        for (const auto& p : points) {
            if (p.id >= 0) {
                counts[p.id]++;
                if (p.id > max_id_final) max_id_final = p.id;
            }
        }
        
        if (counts.size() < MIN_REQUIRED_IDS) {
             while (counts.size() < MIN_REQUIRED_IDS) {
                  int best_pid = -1;
                  int max_size = 0;
                  for(const auto& pair : counts) {
                      if (pair.second > max_size) {
                          max_size = pair.second;
                          best_pid = pair.first;
                      }
                  }
                  
                  if (max_size < 2) break;
                  
                  std::vector<size_t> members;
                  for(size_t i=0; i<points.size(); ++i) {
                      if(points[i].id == best_pid) members.push_back(i);
                  }
                  
                  double min_x = std::numeric_limits<double>::max();
                  double max_x = std::numeric_limits<double>::lowest();
                  double min_y = std::numeric_limits<double>::max();
                  double max_y = std::numeric_limits<double>::lowest();

                  for(size_t idx : members) { 
                      if(points[idx].x < min_x) min_x = points[idx].x;
                      if(points[idx].x > max_x) max_x = points[idx].x;
                      if(points[idx].y < min_y) min_y = points[idx].y;
                      if(points[idx].y > max_y) max_y = points[idx].y;
                  }

                  bool split_by_x = (max_x - min_x) > (max_y - min_y);
                  
                  std::sort(members.begin(), members.end(), [&](size_t a, size_t b) {
                       if (split_by_x) return points[a].x < points[b].x;
                       return points[a].y < points[b].y;
                  });
                  
                  int new_id = ++max_id_final;
                  size_t split_start = members.size() / 2;
                  for(size_t k = split_start; k < members.size(); ++k) {
                       points[members[k]].id = new_id;
                  }
                  
                  counts[best_pid] = split_start;
                  counts[new_id] = members.size() - split_start;
             }
             return;
        }
        
        if (counts.size() < MIN_TOTAL_IDS_THRESHOLD) return;

        for (size_t i = 0; i < points.size(); ++i) {
            int pid = points[i].id;
            if (pid < 0) continue;
            
            // Check count
            if (counts[pid] < SMALL_CLUSTER_THRESHOLD) {
                // Find nearest neighbor
                double min_dist = MAX_MERGE_DIST_SQ;
                int best_id = -1;
                
                for (size_t j = 0; j < points.size(); ++j) {
                     if (points[j].id == pid || points[j].id < 0) continue;
                     
                     // Distance check
                     double dist = (points[i].x - points[j].x)*(points[i].x - points[j].x) + 
                                   (points[i].y - points[j].y)*(points[i].y - points[j].y);
                     
                     if (dist < min_dist) {
                         min_dist = dist;
                         best_id = points[j].id;
                     }
                }
                
                if (best_id != -1) {
                    // Update
                    counts[pid]--;
                    points[i].id = best_id;
                    counts[best_id]++;
                }
            }
        }
    }
}

void VisualIntegrity::saveDebugImage(const FramePtr& frame, const PointMap& landmarks_map, const std::string& filename) 
{
    // Ensure we have an image
    if (frame->img_pyr_.empty()) return;

    cv::Mat img_color;
    // Check if grayscale
    if (frame->img_pyr_[0].channels() == 1) {
        cv::cvtColor(frame->img_pyr_[0], img_color, cv::COLOR_GRAY2BGR);
    } else {
        img_color = frame->img_pyr_[0].clone();
    }

    // Prepare to count features per Object ID
    std::map<int, int> id_counts;
    
    // Quick lookup for landmarks
    // Let's iterate over landmarks_map to find which features in THIS frame are used.
    std::vector<bool> is_landmark_obs(frame->numFeatures(), false);
    std::vector<BalancePoint> points_to_balance;
    std::vector<size_t> feat_indices_for_points;
    
    for (const auto& kv : landmarks_map) {
        const auto& landmark = kv.second;
        // Check observations
        for (const auto& obs : landmark.observations) {
            // obs.first is FrameID (BackendId), obs.second is Keypoint Index (uint64_t)
            // We need to match frame->id() with obs.first
            if (obs.first.frame_id == frame->id()) {
                // obs.second is the feature index in the frame
                size_t feat_idx = obs.first.keypoint_index_;
                if (feat_idx < is_landmark_obs.size()) {
                    is_landmark_obs[feat_idx] = true;
                    
                    int obj_id_curr = -1;
                    if (feat_idx < frame->object_id_vec_.size()) {
                        obj_id_curr = frame->object_id_vec_[feat_idx];
                    }
                    points_to_balance.push_back({frame->px_vec_(0, feat_idx), frame->px_vec_(1, feat_idx), obj_id_curr});
                    feat_indices_for_points.push_back(feat_idx);
                }
            }
        }
    }
    
    // Balance IDs
    balanceObjectIds(points_to_balance);

    // Create lookup for new IDs
    std::map<size_t, int> balanced_id_map;
    for (size_t i = 0; i < points_to_balance.size(); ++i) {
        balanced_id_map[feat_indices_for_points[i]] = points_to_balance[i].id;
    }

    for (size_t k = 0; k < frame->numFeatures(); ++k) {
        cv::Point pt(static_cast<int>(frame->px_vec_(0, k)), static_cast<int>(frame->px_vec_(1, k)));
        
        // Color: Red (BGR: 0, 0, 255) if landmark observation, else Black (0, 0, 0)
        cv::Scalar color = is_landmark_obs[k] ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 0);
        
        // Draw point
        cv::circle(img_color, pt, 2, color, -1); // Filled circle
        
        // Count ID
        int obj_id = frame->object_id_vec_[k];
        int final_id = obj_id;

        // Check for balanced ID
        std::string label = "";
        
        if (balanced_id_map.find(k) != balanced_id_map.end()) {
            final_id = balanced_id_map[k];
            if (final_id != obj_id) {
                label = std::to_string(obj_id) + "->" + std::to_string(final_id);
            } else {
                label = std::to_string(obj_id);
            }
        } else {
             if (obj_id >= 0) label = std::to_string(obj_id);
        }
        
        // Draw ID text
        if (!label.empty()) {
            // Tiny text offset
            cv::putText(img_color, label, pt + cv::Point(3, -3), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }

        if (is_landmark_obs[k]) {
            id_counts[final_id]++;
        }
    }


    int y_offset = 20;
    for (const auto& pair : id_counts) {
        std::string info = "ID " + std::to_string(pair.first) + ": " + std::to_string(pair.second);
        
        // Draw legend on top-left
        cv::putText(img_color, info, cv::Point(10, y_offset), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        y_offset += 25;
    }
    
    static int debug_save_cnt = 0;
    cv::imwrite(filename, img_color);
    LOG(INFO) << "Saved debug image to: " << filename;
    
}

bool VisualIntegrity::prepareLinearSystem(const FramePtr& frame, 
                             const std::deque<State>& states, 
                             const size_t state_index, 
                             const Graph* graph, 
                             const PointMap& landmarks_map,
                             const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs,
                             Eigen::MatrixXd& J_all, 
                             Eigen::VectorXd& r_all, 
                             Eigen::MatrixXd&  sig2_int,
                             Eigen::MatrixXd&  sig2_acc,
                             std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                             std::map<uint64_t, std::vector<int>>& lm_to_J_rows,
                             std::map<uint64_t, int>& curr_lm_to_object_ids,
                             std::map<uint64_t, int>& lm_to_object_ids,
                             std::map<std::string, std::vector<int>>& curr_sat_to_J_rows,
                             std::map<std::string, std::vector<int>>& sat_to_J_rows,
                             std::map<uint64_t, std::vector<int>>& curr_imu_to_J_rows,
                             std::map<uint64_t, std::vector<int>>& imu_to_J_rows,
                             std::vector<int>& curr_pose_J_cols)
{
    State state = states[state_index];

    std::vector<std::pair<uint64_t, std::string>> row_ids_all;          // Pair of (ID, Type) for each row in J_all and r_all
    std::vector<std::pair<uint64_t, std::string>> col_ids_all;          // Pair of (ID, Type) for each column in J_all
    std::vector<std::pair<uint64_t, double>> pose_timestamps;           // Pair of (Pose ID, Timestamp) for all poses (cPose, gPose) in the graph
    std::vector<std::pair<uint64_t, int>> rows_curr;                    // Pair of (ID, Type) for rows related to the current state
    std::vector<std::pair<uint64_t, int>> cols_curr;                    // Pair of (ID, Type) for columns related to the current state
    std::map<uint64_t, std::vector<std::string>> gnss_resid_to_prns;
    std::map<uint64_t, std::vector<uint64_t>> gnss_resid_to_param_ids;
    std::map<uint64_t, std::vector<uint64_t>> imu_resid_to_param_ids;

    if (!extractFullLinearSystem(states, state_index, graph, landmarks_map, gnss_measurement_pairs,
                             J_all, r_all, sig2_int, sig2_acc, row_ids_all, col_ids_all, pose_timestamps, rows_curr, cols_curr,
                             gnss_resid_to_prns, gnss_resid_to_param_ids,
                             imu_resid_to_param_ids)) {
        LOG(ERROR) << "Failed to extract linear system.";
        return false;
    }
    std::string debug_dir = "/home/syl/GICI-IM/results/jacobian/";
    printJacobianInfo(J_all, r_all, row_ids_all, col_ids_all, rows_curr, cols_curr, pose_timestamps, debug_dir + "jacobian_visualization"  + std::to_string(state.timestamp) + ".txt");
    // saveEigenMatrixToFile(sig2_int, debug_dir + "sig2_int_output" + std::to_string(state.timestamp)  + ".txt");

    extractLandmarkRowsCols(frame, landmarks_map, row_ids_all, col_ids_all, cols_curr,
                            curr_lm_to_J_rows, lm_to_J_rows, curr_lm_to_object_ids, lm_to_object_ids);
    extractGnssRowsCols(row_ids_all, col_ids_all, cols_curr, gnss_resid_to_prns, gnss_resid_to_param_ids,
                            curr_sat_to_J_rows, sat_to_J_rows);
    extractImuRowsCols(row_ids_all, col_ids_all, cols_curr, imu_resid_to_param_ids,
                       curr_imu_to_J_rows, imu_to_J_rows);
    
    std::map<uint64_t, std::vector<int>> pose_J_cols_dumppy;
    extractPoseRelatedRowsCols(state.id.asInteger(), state.id.type(), cols_curr, pose_J_cols_dumppy, curr_pose_J_cols);

    // CHECK(curr_imu_to_J_rows.size() == 1 && curr_imu_to_J_rows.begin()->second.size() == 15)
    //     << "Expected exactly one IMU residual with 15 columns, but found " << curr_imu_to_J_rows.size() << " residuals and " << (curr_imu_to_J_rows.empty() ? 0 : curr_imu_to_J_rows.begin()->second.size()) << " columns.";
    CHECK(curr_lm_to_J_rows.size() + curr_sat_to_J_rows.size() > 0)
        << "Expected at least one landmark or GNSS residual, but found none.";



    // Debug output
    #if 0
    std::string debug_dir = "/home/syl/GICI-IM/results/debug/";
    saveEigenMatrixToFile(J_all, debug_dir + "J_all_output" + std::to_string(state.timestamp)  + ".txt");
    // saveEigenMatrixToFile(r_all, debug_dir + "r_all_output" + std::to_string(state.timestamp)  + ".txt");
    saveEigenMatrixToFile(sig2_int, debug_dir + "sig2_int_output" + std::to_string(state.timestamp)  + ".txt");
    // saveEigenMatrixToFile(sig2_acc, debug_dir + "sig2_acc_output" + std::to_string(state.timestamp)  + ".txt");
    // saveFactorGraphDot(graph, state.id.asInteger(), pose_timestamps, debug_dir + "factor_graph"  + std::to_string(state.timestamp) + ".dot");
    printJacobianInfo(J_all, r_all, row_ids_all, col_ids_all, rows_curr, cols_curr, pose_timestamps, debug_dir + "jacobian_visualization"  + std::to_string(state.timestamp) + ".txt");


    if (frame != nullptr){
        // saveDebugImage(frame, landmarks_map, "/media/syl/longlong/GICI-Dataset/2.1/super/segement_images/" + std::to_string(frame->getTimestampSec()) + "_ids.png");
        saveMeasDebugFile(debug_dir + "landmark_curr_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
            frame->timestamp_/1e9, curr_lm_to_J_rows, "landmark_id", "landmark jacobian shape", &curr_lm_to_object_ids);
        saveMeasDebugFile(debug_dir + "landmark_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
            frame->timestamp_/1e9, lm_to_J_rows, "landmark_id", "landmark jacobian shape", &lm_to_object_ids);
    }

    saveMeasDebugFile(debug_dir + "gnss_sat_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
        gnss_measurement_pairs->back().first.timestamp, sat_to_J_rows, "PRN", "GNSS sat jacobian shape");
    saveMeasDebugFile(debug_dir + "gnss_curr_sat_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
        gnss_measurement_pairs->back().first.timestamp, curr_sat_to_J_rows, "PRN", "GNSS sat jacobian shape");
    saveMeasDebugFile(debug_dir + "imu_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
        state.timestamp, imu_to_J_rows, "imu_residual_id", "IMU jacobian shape");
    saveMeasDebugFile(debug_dir + "imu_curr_jacobian_shape_" + std::to_string(state.timestamp) + ".txt",
        state.timestamp, curr_imu_to_J_rows, "imu_residual_id", "IMU jacobian shape");
    #endif

    return true;
}

bool VisualIntegrity::computeIntegrityMetrics(const Eigen::MatrixXd& J_all,
                                 const Eigen::VectorXd& r_all,
                                 const Eigen::MatrixXd& sig2_int,
                                 const Eigen::MatrixXd& sig2_acc,
                                 const std::map<uint64_t, std::vector<int>>& lm_to_J_rows, // all_lm or curr_lm
                                 const std::map<uint64_t, int>& lm_to_object_ids,
                                 const std::map<std::string, std::vector<int>>& sat_to_J_rows,
                                 const std::map<uint64_t, std::vector<int>>& imu_to_J_rows,
                                 const std::vector<int>& curr_pose_J_cols)
{
    subsets_.clear();
    pap_subset_.clear();
    p_not_monitored_ = 0;

    if (J_all.rows() < 6 || curr_pose_J_cols.size() < 3) {
        LOG(ERROR) << "Not enough data for integrity metrics. rows=" << J_all.rows() << ", pose cols=" << curr_pose_J_cols.size();
        return false;
    }

    // 1. Build fault groups over all sources; each group maps to rows removed by that fault.
    std::vector<std::vector<int>> fault_group_rows;
    std::vector<double> p_prior_groups;
    fault_group_source_ids_.clear();

    // 1.1 Visual fault groups.
    std::vector<uint64_t> curr_lm_ids;
    curr_lm_ids.reserve(lm_to_J_rows.size());
    for (const auto& lm_rows : lm_to_J_rows) {
        curr_lm_ids.push_back(lm_rows.first);
    }

    if (options_.use_segment && curr_lm_ids.size() > 0) {
        std::map<int, std::vector<uint64_t>> object_groups;
        std::vector<uint64_t> independent_lms;
        for (uint64_t lm_id : curr_lm_ids) {
            int obj_id = -1;
            auto obj_it = lm_to_object_ids.find(lm_id);
            if (obj_it != lm_to_object_ids.end()) {
                obj_id = obj_it->second;
            }
            if (obj_id >= 0) {
                object_groups[obj_id].push_back(lm_id);
            } else {
                independent_lms.push_back(lm_id);
            }
        }

        for (const auto& pair : object_groups) {
            std::vector<int> rows;
            for (uint64_t lm_id : pair.second) {
                auto it = lm_to_J_rows.find(lm_id);
                if (it != lm_to_J_rows.end()) {
                    rows.insert(rows.end(), it->second.begin(), it->second.end());
                }
            }
            const double p_group = 1.0 - std::pow(1.0 - options_.visual_prior_fault_probability, static_cast<int>(pair.second.size()));
            addFaultGroup(rows, p_group, "VIS_OBJ:" + std::to_string(pair.first),
                          fault_group_rows, p_prior_groups, fault_group_source_ids_);
        }

        for (uint64_t lm_id : independent_lms) {
            auto it = lm_to_J_rows.find(lm_id);
            if (it != lm_to_J_rows.end()) {
                addFaultGroup(it->second, options_.visual_prior_fault_probability,
                              "VIS_LM:" + std::to_string(lm_id),
                              fault_group_rows, p_prior_groups, fault_group_source_ids_);
            }
        }
    } else {
        for (uint64_t lm_id : curr_lm_ids) {
            auto it = lm_to_J_rows.find(lm_id);
            if (it != lm_to_J_rows.end()) {
                addFaultGroup(it->second, options_.visual_prior_fault_probability,
                              "VIS_LM:" + std::to_string(lm_id),
                              fault_group_rows, p_prior_groups, fault_group_source_ids_);
            }
        }
    }

    // 1.2 GNSS fault groups: satellite fault, constellation common fault, baseline fault.
    std::map<int, std::vector<int>> constellation_rows;
    std::vector<int> gnss_all_rows;
    for (const auto& sat_pair : sat_to_J_rows) {
        const std::string& prn = sat_pair.first;
        const std::vector<int>& rows = sat_pair.second;
        if (rows.empty()) continue;
        const int sys_idx = getGnssSystemIndex(prn);
        const double p_sat = getBoundedProbabilityFromVector(options_.gnss_sat_prior_fault_probability, sys_idx, 1.0e-8);
        addFaultGroup(rows, p_sat, "GNSS_SAT:" + prn,
                  fault_group_rows, p_prior_groups, fault_group_source_ids_);

        auto& sys_rows = constellation_rows[sys_idx];
        sys_rows.insert(sys_rows.end(), rows.begin(), rows.end());
        gnss_all_rows.insert(gnss_all_rows.end(), rows.begin(), rows.end());
    }

    for (const auto& sys_pair : constellation_rows) {
        const int sys_idx = sys_pair.first;
        std::vector<double> p_const_or_ref = options_.gnss_const_prior_fault_probability;
        for (int i = 0; i < p_const_or_ref.size(); ++i) {
            if (options_.gnss_sat_prior_fault_probability[i] > p_const_or_ref[i]) p_const_or_ref[i] = options_.gnss_sat_prior_fault_probability[i];
        }
        const double p_const = getBoundedProbabilityFromVector(p_const_or_ref, sys_idx, 1.0e-8);
        addFaultGroup(sys_pair.second, p_const, "GNSS_CONST/REF_SAT:" + std::to_string(sys_idx),
                      fault_group_rows, p_prior_groups, fault_group_source_ids_);
    }

    // // In order to decrease the computation, we think ref fault and constellation fault together, and use the same prior. 
    // for (const auto& sys_pair : constellation_rows) {
    //     const int sys_idx = sys_pair.first;
    //     const std::vector<int>& sys_rows = sys_pair.second;
    //     if (sys_rows.empty()) {
    //         continue;
    //     }

    //     std::string ref_prn_for_group = "UNKNOWN";
    //     size_t max_row_count = 0;
    //     for (const auto& sat_pair : sat_to_J_rows) {
    //         const std::string& prn = sat_pair.first;
    //         if (getGnssSystemIndex(prn) != sys_idx) {
    //             continue;
    //         }
    //         if (sat_pair.second.size() > max_row_count) {
    //             max_row_count = sat_pair.second.size();
    //             ref_prn_for_group = prn;
    //         }
    //     }

    //     const double ref_p = getBoundedProbabilityFromVector(
    //         options_.gnss_sat_prior_fault_probability, sys_idx, 1.0e-8);
    //     addFaultGroup(sys_rows, ref_p,
    //                   "GNSS_SAT_REF_SYS:" + std::to_string(sys_idx) + ":" + ref_prn_for_group,
    //                   fault_group_rows, p_prior_groups, fault_group_source_ids_);
    // }

    // 1.3 INS fault groups: each IMU residual is an independent fault source.
    for (const auto& imu_pair : imu_to_J_rows) {
        addFaultGroup(imu_pair.second, options_.imu_prior_fault_probability,
                      "IMU_RESID:" + std::to_string(imu_pair.first),
                      fault_group_rows, p_prior_groups, fault_group_source_ids_);
    }

    if (fault_group_rows.empty()) {
        LOG(ERROR) << "No fault groups were constructed for integrity monitoring.";
        return false;
    }
    CHECK(fault_group_rows.size() == p_prior_groups.size() && fault_group_rows.size() == fault_group_source_ids_.size())
        << "Mismatch in fault group data structures sizes.";

    #if 1
    // Debug output of fault groups
    std::string debug_dir = "/home/syl/GICI-IM/results/debug/";
    std::ofstream fg_out(debug_dir + "fault_groups_" + std::to_string(timestamp_) + ".txt");
    if (fg_out.is_open()) {
        for (size_t i = 0; i < fault_group_rows.size(); ++i) {
            fg_out << "Group " << i << ", Source: " << fault_group_source_ids_[i] << ", Prior P: " << std::fixed << std::setprecision(9) << p_prior_groups[i] << ", Rows: ";
            for (int row : fault_group_rows[i]) {
                fg_out << row << " ";            
            }
            fg_out << "\n";
        }
        fg_out.close();
        LOG(INFO) << "Saved fault group details to " << debug_dir + "fault_groups_" + std::to_string(timestamp_) + ".txt";
    }
    #endif

    // 2. Determine subsets from expanded prior fault groups.
    determineSubsets(p_prior_groups, subsets_, pap_subset_, p_not_monitored_);
    if (subsets_.empty()) {
        LOG(ERROR) << "Failed to determine subsets from fault priors.";
        return false;
    }
    CHECK_EQ(fault_group_rows.size(), subsets_[0].size());
    CHECK_EQ(fault_group_rows.size(), fault_group_source_ids_.size());

    LOG(INFO) << "Total fault groups: " << fault_group_rows.size()
              << ", total subsets to monitor: " << subsets_.size();

    // 3. Compute subset solutions for all monitored fault hypotheses.
    computeSubsetSolution(J_all, r_all, sig2_int, sig2_acc,
                          subsets_, fault_group_rows, curr_pose_J_cols,
                          sigma_, bias_, sigma_ss_, bias_ss_, s1vec_, s2vec_, s3vec_, x_, chi2_);

    // 4. Filter unmonitorable subsets and continue legacy threshold/PL logic.
    filteroutSubsets(sigma_, bias_, sigma_ss_, bias_ss_, s1vec_, s2vec_, s3vec_, x_, chi2_, subsets_, pap_subset_, p_not_monitored_);

    // 6. Compute Test Thresholds
    T_ = computeTestThresholds(sigma_ss_, bias_ss_);

    // 7. Fault Detection
    bool fault_detected = false;
    int fault_detected_num = 0;
    for (int i = 0; i < T_.rows(); ++i) {
        for (int q = 0; q < 3; ++q) {
            double test_stat = std::abs(x_(i, q) - x_(0, q));
            if (test_stat > T_(i, q)) {
                fault_detected = true;
                fault_detected_num++;
                break;
                // LOG(WARNING) << "Fault detected in subset " << i << " axis " << q << std::endl;
            }
        }
    }

    // saveEigenMatrixToFile(sig2_int, "/home/syl/GICI-IM/results/debug/sig2_int_" + std::to_string(timestamp_) + ".txt");
    // saveEigenMatrixToFile(sigma_, "/home/syl/GICI-IM/results/debug/sigma_" + std::to_string(timestamp_) + ".txt");
    // saveEigenMatrixToFile(sigma_ss_, "/home/syl/GICI-IM/results/debug/sigma_ss_" + std::to_string(timestamp_) + ".txt");
    // saveEigenMatrixToFile(T_, "/home/syl/GICI-IM/results/debug/T_" + std::to_string(timestamp_) + ".txt");
    // saveEigenMatrixToFile(x_, "/home/syl/GICI-IM/results/debug/x_" + std::to_string(timestamp_) + ".txt");
    if (fault_detected) LOG(WARNING) << std::fixed << std::setprecision(6)<< "Fault detected num: " << fault_detected_num << ", for timestamp: " << timestamp_;
    if (!fault_detected)  LOG(INFO) << std::fixed << std::setprecision(6)<< "No fault detected for timestamp: " << timestamp_;

    // 8. Compute PL and IR
    computePL(sigma_, bias_, T_, pap_subset_, p_not_monitored_, VPL_, LaPL_, LoPL_, HPL_);

    IR_ = computeIR(sigma_, bias_, T_, pap_subset_, p_not_monitored_);

    // return fault_detected;
    return true;
}

int VisualIntegrity::getGnssSystemIndex(const std::string& prn) const
{
    if (prn.empty()) return 0;
    const char sys = static_cast<char>(std::toupper(static_cast<unsigned char>(prn.front())));
    switch (sys) { // [GPS, GLO, BDS, GAL]
        case 'G': return 0;
        case 'R': return 1;
        case 'C': return 2;
        case 'E': return 3;
        default: return 0;
    }
}

double VisualIntegrity::getBoundedProbabilityFromVector(const std::vector<double>& probs, int idx, double fallback) const
{
    double p = fallback;
    if (idx >= 0 && idx < static_cast<int>(probs.size())) {
        p = probs[idx];
    } else if (!probs.empty()) {
        p = probs.front();
    }
    return std::min(1.0 - 1.0e-12, std::max(1.0e-12, p));
}


void VisualIntegrity::addFaultGroup(const std::vector<int>& rows_in, const double p_in, const std::string& source_id,
                                    std::vector<std::vector<int>>& rows_groups, std::vector<double>& p_groups,
                                    std::vector<std::string>& source_ids) const
{
    std::vector<int> rows = rows_in;
    if (rows.empty()) return;
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    if (rows.empty()) return;
    rows_groups.push_back(rows);
    p_groups.push_back(std::min(1.0 - 1.0e-12, std::max(1.0e-12, p_in)));
    source_ids.push_back(source_id);
}


double VisualIntegrity::getGnssSystemFactor(const std::string& prn) const
{
    auto get_factor = [&](size_t idx) -> double {
        if (idx < options_.user_F.size()) {
            return options_.user_F[idx];
        }
        if (!options_.user_F.empty()) {
            return options_.user_F.front();
        }
        return 1.0;
    };

    if (prn.empty()) return get_factor(0);
    const char sys = static_cast<char>(std::toupper(static_cast<unsigned char>(prn.front())));
    switch (sys) {
        case 'G': return get_factor(0);
        case 'R': return get_factor(1);
        case 'C': return get_factor(2);
        case 'E': return get_factor(3);
        default: return get_factor(0);
    }
}
// F^s^2 R_r^2 (a_σ^2+(b_σ^2)/sin^2⁡(θ_r^s ) )
double VisualIntegrity::computeGnssUserSigma(const double elevation, const std::string& prn) const
{
    const double user_F = getGnssSystemFactor(prn);
    const double sin_el = std::max(1.0e-3, std::sin(std::max(1.0e-3, elevation)));
    const double sigma_sq = user_F * user_F * options_.user_Rr * options_.user_Rr *
                            (options_.user_a_sigma * options_.user_a_sigma +
                             (options_.user_b_sigma * options_.user_b_sigma) / (sin_el * sin_el));
    return std::sqrt(std::max(1.0e-12, sigma_sq));
}

double VisualIntegrity::computeGnssDopplerSigma(const std::string& prn) const
{
    const double user_F = getGnssSystemFactor(prn);
    const double sigma_sq = user_F * user_F * options_.doppler_c_sigma * options_.doppler_c_sigma;
    return std::sqrt(std::max(1.0e-12, sigma_sq));
}


double VisualIntegrity::computeGnssSpatialCorrelation(double az1, double el1, double az2, double el2) const
{
    const double cos_delta = std::sin(el1) * std::sin(el2) +
                             std::cos(el1) * std::cos(el2) * std::cos(az1 - az2);
    const double bounded = std::max(-1.0, std::min(1.0, cos_delta));
    const double dpsi = std::acos(bounded);
    return options_.rho_max * std::exp(-dpsi / std::max(1.0e-6, (options_.psi_user_deg * D2R)));
}

double VisualIntegrity::computeGnssCodeDdVariance(double sigma_user, double sigma_ref, double rho_sr) const
{
    return 2.0 * (sigma_user * sigma_user + sigma_ref * sigma_ref -
                  2.0 * sigma_user * sigma_ref * rho_sr);
}



bool VisualIntegrity::updateGnssSatelliteInfo(const GnssMeasurement& measurement,
                                              const std::string& prn,
                                              const std::string& ref_prn,
                                              std::string& out_prn,
                                              std::string& out_ref_prn,
                                              double& out_elevation,
                                              double& out_azimuth,
                                              double& out_ref_elevation,
                                              double& out_ref_azimuth) const
{
    out_prn = prn;
    out_ref_prn = ref_prn;

    auto sat_it = measurement.satellites.find(prn);
    if (sat_it == measurement.satellites.end()) {
        return false;
    }

    const auto& sat = sat_it->second;
    out_elevation = gnss_common::satelliteElevation(sat.sat_position, measurement.position);
    out_azimuth = gnss_common::satelliteAzimuth(sat.sat_position, measurement.position);

    if (!ref_prn.empty()) {
        auto sat_ref_it = measurement.satellites.find(ref_prn);
        if (sat_ref_it != measurement.satellites.end()) {
            const auto& sat_ref = sat_ref_it->second;
            out_ref_elevation = gnss_common::satelliteElevation(sat_ref.sat_position, measurement.position);
            out_ref_azimuth = gnss_common::satelliteAzimuth(sat_ref.sat_position, measurement.position);
            return true;
        }
    }

    out_ref_prn.clear();
    out_ref_elevation = out_elevation;
    out_ref_azimuth = out_azimuth;
    return true;
}


bool VisualIntegrity::extractFullLinearSystem(const std::deque<State>& states, const size_t state_index, const Graph* graph, const PointMap& landmarks_map,
                                              const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs,
                                              Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::MatrixXd& sig2_int, Eigen::MatrixXd& sig2_acc,
                                              std::vector<std::pair<uint64_t, std::string>>& row_ids_all, std::vector<std::pair<uint64_t, std::string>>& col_ids_all, std::vector<std::pair<uint64_t, double>>& pose_timestamps,
                                              std::vector<std::pair<uint64_t, int>>& rows_curr, std::vector<std::pair<uint64_t, int>>& cols_curr,
                                              std::map<uint64_t, std::vector<std::string>>& gnss_resid_to_prns,
                                              std::map<uint64_t, std::vector<uint64_t>>& gnss_resid_to_param_ids,
                                              std::map<uint64_t, std::vector<uint64_t>>& imu_resid_to_param_ids
                                              )
{
    
    
    
    const State& current_state = states[state_index];
    const uint64_t current_pose_id = states[state_index].id.asInteger();
    const IdType current_pose_type = states[state_index].id.type();
    if (!graph->parameterBlockExists(current_pose_id)) return false;

    std::vector<GenericResidualInfo> all_residuals;

    ceres::Problem* problem = graph->problem().get();
    std::vector<ceres::ResidualBlockId> residual_blocks;
    problem->GetResidualBlocks(&residual_blocks);

    for (auto residual_block_id : residual_blocks) {
        const ceres::CostFunction* cost_function = problem->GetCostFunctionForResidualBlock(residual_block_id);
        if (cost_function == nullptr) continue;

        int num_residuals = cost_function->num_residuals();
        gici::Graph::ParameterBlockCollection parameter_blocks = graph->parameters(residual_block_id);
        
        std::vector<double> residuals_eval(num_residuals);
        std::vector<double*> jacobians(parameter_blocks.size());
        std::vector<std::vector<double>> jacobian_buffers(parameter_blocks.size());
        std::vector<double*> parameter_blocks_ptrs;
        problem->GetParameterBlocksForResidualBlock(residual_block_id, &parameter_blocks_ptrs);

        uint64_t row_id = 0;
        bool is_current = false;
        double timestamp = -1.0;

        for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            BackendId pb_id(parameter_blocks[i].first);
            int param_dim = parameter_blocks[i].second->minimalDimension();
            
            jacobian_buffers[i].resize(num_residuals * param_dim);
            jacobians[i] = jacobian_buffers[i].data();

            if (problem->IsParameterBlockConstant(parameter_blocks_ptrs[i])) {
                jacobians[i] = nullptr; 
            }

            if (pb_id.asInteger() == current_pose_id) {
                is_current = true;
            }
            // Check for IMU states associated with current pose
            if (pb_id.type() == IdType::ImuStates) {
                 BackendId pose_id = changeIdType(pb_id, current_pose_type);
                 if (pose_id.asInteger() == current_pose_id) {
                    is_current = true;
                 }
            }

            
            for (const auto& state : states) {
                if (state.id.asInteger() == pb_id.asInteger()) {
                    timestamp = state.timestamp;
                    if (pb_id.type() == IdType::cPose || pb_id.type() == IdType::gPose) {
                        bool exists = false;
                        for (const auto& pt : pose_timestamps) {
                            if (pt.first == pb_id.asInteger()) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) {
                            pose_timestamps.push_back({pb_id.asInteger(), timestamp});
                        }
                    }
                    break;
                }
            }

        }

        if (!problem->EvaluateResidualBlock(residual_block_id, false, nullptr, residuals_eval.data(), jacobians.data())) {
            continue;
        }

        const ErrorInterface* base_err = graph->errorInterfacePtr(residual_block_id).get();
        if (base_err == nullptr) continue;

        if (num_residuals > 0) {
            Eigen::VectorXd residual_unweighted =
                Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals);
            base_err->deNormalizeResidual(residual_unweighted.data());
            Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals) = residual_unweighted;

            Eigen::MatrixXd sqrt_info_inv = Eigen::MatrixXd::Zero(num_residuals, num_residuals);
            for (int col = 0; col < num_residuals; ++col) {
                Eigen::VectorXd basis = Eigen::VectorXd::Zero(num_residuals);
                basis(col) = 1.0;
                base_err->deNormalizeResidual(basis.data());
                sqrt_info_inv.col(col) = basis;
            }

            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                if (jacobians[i] == nullptr) {
                    continue;
                }
                int param_dim = parameter_blocks[i].second->minimalDimension();
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    J_weighted(jacobians[i], num_residuals, param_dim);
                const Eigen::MatrixXd J_unweighted = sqrt_info_inv * J_weighted;
                J_weighted = J_unweighted;
            }    
        }

        auto error_type = base_err->typeInfo();
        GenericResidualInfo info;
        info.error_type_str = kErrorToStr.at(error_type);
        info.timestamp = timestamp;
        info.row_id = {reinterpret_cast<uint64_t>(residual_block_id), info.error_type_str};
        info.residual = Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals);
        info.is_current_frame = is_current;

        for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            if (jacobians[i] != nullptr) {
                int dim = parameter_blocks[i].second->minimalDimension();
                Eigen::MatrixXd J = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[i], num_residuals, dim);
                info.jacobians.push_back({parameter_blocks[i].first, J});
            }
        }

        info.sig2_int = 1.0; // Default
        info.sig2_acc = 1.0; // Default
        info.landmark_id = 0; // Default
        info.cur_track = -1;
        info.prn = "";
        if (error_type == ErrorType::kReprojectionError) {
            if (options_.use_complex_visual_cov && (options_.overbounding_func != "none" || options_.normal_func != "none")) {
                // Find landmark ID from parameter blocks
                for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                    BackendId pb_id(parameter_blocks[i].first);
                    if (pb_id.type() == IdType::cLandmark) {
                        info.landmark_id = pb_id.asInteger();
                        break;
                    }
                }
                if (info.landmark_id != 0) {
                    info.sig2_int = options_.sigma_pixel * options_.sigma_pixel;
                    info.sig2_acc = info.sig2_int;
                }
            } else {
                info.landmark_id = 0;
                info.sig2_int = options_.simple_visual_sigma * options_.simple_visual_sigma;
                info.sig2_acc = info.sig2_int;
            }

        }
        else if (error_type == ErrorType::kPseudorangeErrorDD ||
                 error_type == ErrorType::kPhaserangeErrorDD ||
                 error_type == ErrorType::kDopplerError) {
            const GnssMeasurement* gnss_measurement_for_residual = selectGnssMeasurementForTimestamp(info.timestamp, gnss_measurement_pairs);
            bool valid_meta = false;
            if (gnss_measurement_for_residual != nullptr) {
                if (error_type == ErrorType::kPseudorangeErrorDD) {
                    info.gnss_kind = GnssKind::CodeDD;
                    GnssMeasurementDDIndexPair dd_index;
                    bool found = false;
                    if (auto e = dynamic_cast<const PseudorangeErrorDD<3>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PseudorangeErrorDD<7,3>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PseudorangeErrorDD<3,1,1,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PseudorangeErrorDD<7,3,1,1,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    if (found) {
                        valid_meta = updateGnssSatelliteInfo(*gnss_measurement_for_residual, dd_index.rov.prn, dd_index.rov_base.prn,
                                                                info.prn, info.ref_prn,
                                                                info.elevation, info.azimuth, info.ref_elevation,info.ref_azimuth);
                        if (valid_meta) {
                            info.sigma_user = computeGnssUserSigma(info.elevation, info.prn);
                            if (!info.ref_prn.empty()) info.sigma_user_ref = computeGnssUserSigma(info.ref_elevation, info.ref_prn);
                        }
                    }
                } else if (error_type == ErrorType::kPhaserangeErrorDD) {
                    info.gnss_kind = GnssKind::PhaseDD;
                    GnssMeasurementDDIndexPair dd_index;
                    bool found = false;
                    if (auto e = dynamic_cast<const PhaserangeErrorDD<3,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PhaserangeErrorDD<7,3,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PhaserangeErrorDD<3,1,1,1,1,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const PhaserangeErrorDD<7,3,1,1,1,1,1,1>*>(base_err)) { dd_index = e->getGnssMeasurementIndex(); found = true; }
                    if (found) {
                        valid_meta = updateGnssSatelliteInfo(*gnss_measurement_for_residual, dd_index.rov.prn, dd_index.rov_base.prn,
                                                                info.prn, info.ref_prn,
                                                                info.elevation, info.azimuth, info.ref_elevation,info.ref_azimuth);
                        if (valid_meta) {
                            info.sigma_user = computeGnssUserSigma(info.elevation, info.prn);
                            if (!info.ref_prn.empty()) info.sigma_user_ref = computeGnssUserSigma(info.ref_elevation, info.ref_prn);
                        }
                    }
                } else {
                    info.gnss_kind = GnssKind::Doppler;
                    GnssMeasurementIndex idx;
                    bool found = false;
                    if (auto e = dynamic_cast<const DopplerError<3,3,1>*>(base_err)) { idx = e->getGnssMeasurementIndex(); found = true; }
                    else if (auto e = dynamic_cast<const DopplerError<7,9,3,1>*>(base_err)) { idx = e->getGnssMeasurementIndex(); found = true; }
                    if (found) {
                        valid_meta = updateGnssSatelliteInfo(*gnss_measurement_for_residual, idx.prn, "",
                                                                info.prn, info.ref_prn,
                                                                info.elevation, info.azimuth, info.ref_elevation,info.ref_azimuth);
                        if (valid_meta) {
                            info.sigma_user = computeGnssDopplerSigma(info.prn);
                        }
                    }
                }
            }

            if (valid_meta && options_.use_complex_gnss_cov) {
                double var_diag = 1.0;
                if (info.gnss_kind == GnssKind::CodeDD) {
                    const double rho_sr = computeGnssSpatialCorrelation(info.azimuth, info.elevation,
                                                                        info.ref_azimuth, info.ref_elevation);
                    var_diag = computeGnssCodeDdVariance(info.sigma_user, info.sigma_user_ref, rho_sr);
                } else if (info.gnss_kind == GnssKind::PhaseDD) {
                    const double rho_sr = computeGnssSpatialCorrelation(info.azimuth, info.elevation,
                                                                        info.ref_azimuth, info.ref_elevation);
                    var_diag = computeGnssCodeDdVariance(info.sigma_user, info.sigma_user_ref, rho_sr) /
                                (options_.user_Rr * options_.user_Rr);
                } else {
                    var_diag = info.sigma_user;
                }
                info.sig2_int = std::max(1.0e-8, var_diag);
                info.sig2_acc = info.sig2_int;
            } else {
                info.sig2_int = options_.simple_gnss_sigma * options_.simple_gnss_sigma;
                info.sig2_acc = info.sig2_int;
            }
        }
        else if (error_type == ErrorType::kIMUError) {
            auto imu_err = dynamic_cast<const ImuError*>(base_err);
            if (imu_err != nullptr && options_.use_complex_imu_cov) {
                info.is_imu = true;
                info.sig2_imu = imu_err->covarianceMatrix();
                if (info.sig2_imu.rows() != num_residuals || info.sig2_imu.cols() != num_residuals) {
                    LOG(WARNING) << "IMU covariance size mismatch for residual block ";
                    info.sig2_imu = Eigen::MatrixXd::Identity(num_residuals, num_residuals) * options_.simple_imu_sigma * options_.simple_imu_sigma;
                }
            } else {
                info.is_imu = true;
                info.sig2_imu = Eigen::MatrixXd::Identity(num_residuals, num_residuals) * options_.simple_imu_sigma * options_.simple_imu_sigma;
                info.sig2_int = options_.simple_imu_sigma * options_.simple_imu_sigma;
                info.sig2_acc = info.sig2_int;
            }
        }
        else { //other types of residuals
            if (options_.use_complex_others_cov) {
                info.sig2_others = Eigen::MatrixXd::Identity(num_residuals, num_residuals);
                if (base_err == nullptr || !base_err->getCovarianceMatrix(info.sig2_others)) {
                    LOG(WARNING) << "Failed to extract generic covariance for error type: " << info.error_type_str
                                << ", fallback to identity.";
                    info.sig2_others = Eigen::MatrixXd::Identity(num_residuals, num_residuals);
                }

                info.sig2_others = 0.5 * (info.sig2_others + info.sig2_others.transpose())
                                    + 1.0e-10 * Eigen::MatrixXd::Identity(num_residuals, num_residuals);

                if (info.sig2_others.hasNaN() || info.sig2_others.rows() != num_residuals || info.sig2_others.cols() != num_residuals) {
                    LOG(WARNING) << "Invalid generic covariance for error type: " << info.error_type_str << ", fallback to identity.";
                    info.sig2_others = Eigen::MatrixXd::Identity(num_residuals, num_residuals);
                }

                if (info.sig2_others.rows() == num_residuals && info.sig2_others.cols() == num_residuals) {
                    const double avg_diag = std::max(1.0e-8, info.sig2_others.diagonal().mean());
                    info.sig2_int = avg_diag;
                    info.sig2_acc = avg_diag;
                } 
            }
            else {
                info.sig2_others = Eigen::MatrixXd::Identity(num_residuals, num_residuals);
                info.sig2_int = options_.simple_others_sigma * options_.simple_others_sigma;
                info.sig2_acc = info.sig2_int;
            }
        }

        all_residuals.push_back(info);

        if (info.gnss_kind != GnssKind::None && !info.prn.empty()) {
            const uint64_t residual_id = info.row_id.first;
            auto& prn_vec = gnss_resid_to_prns[residual_id];
            prn_vec.push_back(info.prn);
            if (!info.ref_prn.empty() && info.ref_prn != info.prn) {
                prn_vec.push_back(info.ref_prn);
            }

            auto& param_vec = gnss_resid_to_param_ids[residual_id];
            for (const auto& jac_pair : info.jacobians) {
                param_vec.push_back(jac_pair.first);
            }
        }

        if (info.is_imu) {
            const uint64_t residual_id = info.row_id.first;
            auto& param_vec = imu_resid_to_param_ids[residual_id];
            for (const auto& jac_pair : info.jacobians) {
                param_vec.push_back(jac_pair.first);
            }
        }
    }

    for (auto& pair : gnss_resid_to_prns) {
        auto& prns = pair.second;
        std::sort(prns.begin(), prns.end());
        prns.erase(std::unique(prns.begin(), prns.end()), prns.end());
    }
    for (auto& pair : gnss_resid_to_param_ids) {
        auto& params = pair.second;
        std::sort(params.begin(), params.end());
        params.erase(std::unique(params.begin(), params.end()), params.end());
    }
    for (auto& pair : imu_resid_to_param_ids) {
        auto& params = pair.second;
        std::sort(params.begin(), params.end());
        params.erase(std::unique(params.begin(), params.end()), params.end());
    }

    std::sort(all_residuals.begin(), all_residuals.end(), [](const GenericResidualInfo& a, const GenericResidualInfo& b) {
        if (a.jacobians.front().first != b.jacobians.front().first) {
            return a.jacobians.front().first < b.jacobians.front().first;
        }
        return a.row_id.second < b.row_id.second;
    });

    // Build Landmark to Indices Map
    std::map<uint64_t, std::vector<size_t>> lm_to_indices;
    for(size_t i = 0; i < all_residuals.size(); ++i) {
            if(all_residuals[i].landmark_id > 0) lm_to_indices[all_residuals[i].landmark_id].push_back(i);
    }

    // Optimization: Batch compute sig2 overbounding
    if (options_.use_complex_visual_cov && (options_.overbounding_func == "dual_exp" || options_.normal_func == "dual_exp")) {
        
        for(auto const& pair : lm_to_indices) {
            uint64_t lm_id = pair.first;
            if (landmarks_map.find(BackendId(lm_id)) == landmarks_map.end()) continue;
            const auto& landmark = landmarks_map.at(BackendId(lm_id));

            std::map<uint64_t, int> res_to_frame;
            for(const auto& obs : landmark.observations) {
                res_to_frame[obs.second] = obs.first.frame_id;
            }

            std::unordered_map<int, int> frame_to_track_exact;
            int min_frame_id = std::numeric_limits<int>::max();
            for(size_t k = 0; k < landmark.point->obs_.size(); ++k) {
                const int frame_id = landmark.point->obs_[k].frame_id;
                frame_to_track_exact[frame_id] = static_cast<int>(k);
                if (frame_id < min_frame_id) {
                    min_frame_id = frame_id;
                }
            }

            for(size_t idx : pair.second) {
                uint64_t res_id = all_residuals[idx].row_id.first;
                if(res_to_frame.count(res_id)) {
                    int fid = res_to_frame[res_id];
                    if (!frame_to_track_exact.empty()) {
                        auto track_it = frame_to_track_exact.find(fid);
                        int j = 0;
                        while (track_it == frame_to_track_exact.end() && fid > min_frame_id) {
                            --fid;
                            track_it = frame_to_track_exact.find(fid);
                            j++;
                        }
                        if (j > 0) {
                            LOG(INFO) << "For residual block ID " << res_id << ", searched back " << j << " frames to find track for frame " << res_to_frame[res_id];
                        }

                        if (track_it != frame_to_track_exact.end()) {
                            const int cur_track = track_it->second;
                            all_residuals[idx].cur_track = cur_track;
                            if (options_.overbounding_func != "none") all_residuals[idx].sig2_int += computeDualExpOverboundingSig2(options_.overbounding_parameters, 0, cur_track);
                            if (options_.normal_func != "none")  all_residuals[idx].sig2_acc += computeDualExpOverboundingSig2(options_.normal_parameters, 0, cur_track);
                        }
                    }
                }
            }
        }
    }


    
    // Build Column Map
    std::map<uint64_t, int> param_col_map;
    std::map<uint64_t, int> param_dim_map;
    
    for (const auto& res : all_residuals) {
        for (const auto& pair : res.jacobians) {
            if (param_col_map.find(pair.first) == param_col_map.end()) {
                param_col_map[pair.first] = -1;
                param_dim_map[pair.first] = pair.second.cols();
            }
        }
    }

    // Assign columns (sorted by ID, but grouping Pose with IMUStates)
    std::vector<uint64_t> ordered_param_ids;
    std::set<uint64_t> processed_ids;
    
    // Identify Poses and pair with IMU states
    // Sort poses by timestamp
    struct PoseSortInfo {
        uint64_t id;
        double timestamp;
    };
    std::vector<PoseSortInfo> poses_to_sort;

    for (auto const& pair : param_col_map) {
        uint64_t id = pair.first;
        BackendId bid(id);
        if (bid.type() == IdType::cPose || bid.type() == IdType::gPose) {
            double ts = 0.0;
            bool found = false;
            for(const auto& pt : pose_timestamps) {
                if(pt.first == id) {
                    ts = pt.second;
                    found = true;
                    break;
                }
            }
            // If not found in pose_timestamps (e.g. not in states list), use ID as proxy for time
            if(!found) ts = static_cast<double>(id);
            
            poses_to_sort.push_back({id, ts});
        }
    }

    std::sort(poses_to_sort.begin(), poses_to_sort.end(), [](const PoseSortInfo& a, const PoseSortInfo& b){
        return a.timestamp < b.timestamp;
    });

    for (const auto& info : poses_to_sort) {
        uint64_t id = info.id;
        if (processed_ids.count(id)) continue;
        
        ordered_param_ids.push_back(id);
        processed_ids.insert(id);
        
        BackendId bid(id);
        BackendId sb_bid = changeIdType(bid, IdType::ImuStates);
        uint64_t sb_id = sb_bid.asInteger();
        
        if (param_col_map.count(sb_id) && !processed_ids.count(sb_id)) {
            ordered_param_ids.push_back(sb_id);
            processed_ids.insert(sb_id);
        }
    }
    
    // Add remaining parameters (Landmarks, orphan IMU states, etc.)
    for (auto const& pair : param_col_map) {
        uint64_t id = pair.first;
        if (!processed_ids.count(id)) {
            ordered_param_ids.push_back(id);
            processed_ids.insert(id);
        }
    }

    int current_col = 0;
    for (uint64_t id : ordered_param_ids) {
        param_col_map[id] = current_col;
        current_col += param_dim_map[id];
    }

    int N_all_rows = 0;
    for (const auto& res : all_residuals) N_all_rows += res.residual.size();
    int N_all_cols = current_col;

    if (N_all_rows > 0) {
        J_all = Eigen::MatrixXd::Zero(N_all_rows, N_all_cols);
        r_all = Eigen::VectorXd::Zero(N_all_rows); 
        sig2_int = Eigen::MatrixXd::Identity(N_all_rows, N_all_rows);
        sig2_acc = Eigen::MatrixXd::Identity(N_all_rows, N_all_rows);
        row_ids_all.resize(N_all_rows);
        col_ids_all.resize(N_all_cols);

        // Fill Col IDs
        for (auto const& pair : param_col_map) {
            int dim = param_dim_map[pair.first];
            std::string param_id_type = "Unknown";
            BackendId pb_id(pair.first);
            param_id_type = idTypeToString(pb_id.type());
            for (int k = 0; k < dim; ++k) col_ids_all[pair.second + k] = {pair.first, param_id_type};
        }

        // First, set all diagonal elements
        int current_row_idx = 0;
        std::vector<int> res_row_starts;
        res_row_starts.reserve(all_residuals.size());
        
        for (auto& info : all_residuals) {
            res_row_starts.push_back(current_row_idx);
            int num_res = info.residual.size();
            r_all.segment(current_row_idx, num_res) = info.residual;
            
            for (int k = 0; k < num_res; ++k) {
                row_ids_all[current_row_idx + k] = info.row_id;
                sig2_int(current_row_idx + k, current_row_idx + k) = info.sig2_int;
                sig2_acc(current_row_idx + k, current_row_idx + k) = info.sig2_acc;
            }
            if (info.is_imu && options_.use_complex_imu_cov) {
                const int use_n = std::min<int>(num_res, std::min<int>(info.sig2_imu.rows(), info.sig2_imu.cols()));
                if (use_n > 0) {
                    sig2_int.block(current_row_idx, current_row_idx, use_n, use_n) = info.sig2_imu.topLeftCorner(use_n, use_n);
                    sig2_acc.block(current_row_idx, current_row_idx, use_n, use_n) = info.sig2_imu.topLeftCorner(use_n, use_n);
                }
            } else if (info.sig2_others.rows() > 0 && info.sig2_others.cols() > 0 && options_.use_complex_others_cov) {
                const int use_n = std::min<int>(num_res, std::min<int>(info.sig2_others.rows(), info.sig2_others.cols()));
                if (use_n > 0) {
                    sig2_int.block(current_row_idx, current_row_idx, use_n, use_n) = info.sig2_others.topLeftCorner(use_n, use_n);
                    sig2_acc.block(current_row_idx, current_row_idx, use_n, use_n) = info.sig2_others.topLeftCorner(use_n, use_n);
                }
            }

            for (const auto& pair : info.jacobians) { //cols_curr not include imu error to last frame
                int col = param_col_map[pair.first];
                J_all.block(current_row_idx, col, num_res, pair.second.cols()) = pair.second;
                if (info.is_current_frame) {
                    for (int r = 0; r < num_res; ++r) rows_curr.push_back(std::make_pair(info.row_id.first, current_row_idx + r));

                    bool add_curr = false;
                    BackendId parmid(pair.first);
                    if (parmid.type() == IdType::ImuStates) {
                        parmid = changeIdType(parmid, current_pose_type);
                    }
                    if (parmid.asInteger() == current_pose_id || (parmid.type() != IdType::cPose && parmid.type() != IdType::gPose)) {
                        add_curr = true;
                    }
                    if (add_curr) {
                        for (int c = 0; c < pair.second.cols(); ++c) cols_curr.push_back(std::make_pair(pair.first, col + c)); 
                    }
                }
            }
            current_row_idx += num_res;
        }
        
        // Second, set non-diagonal elements based on options
        if (options_.use_complex_visual_cov && (options_.overbounding_func != "none" || options_.normal_func != "none")) {
            // Optimization: Set non-diagonal elements for observations of the same landmark using indexing
            std::vector<uint64_t> valid_lms;
            for(const auto& pair : lm_to_indices) {
                if(pair.second.size() >= 2) valid_lms.push_back(pair.first);
            }

            // #pragma omp parallel for
            for(size_t k = 0; k < valid_lms.size(); ++k) {
                const auto& idxs = lm_to_indices[valid_lms[k]];
                for (size_t i = 0; i < idxs.size(); ++i) {
                    size_t idx1 = idxs[i];
                    auto& ref_info = all_residuals[idx1];
                    int row1 = res_row_starts[idx1];
                    int n1 = ref_info.residual.size();

                    for (size_t j = i + 1; j < idxs.size(); ++j) {
                        size_t idx2 = idxs[j];
                        auto& other_info = all_residuals[idx2];
                        int row2 = res_row_starts[idx2];
                        int n2 = other_info.residual.size();

                        double min_sig2_int = ref_info.sig2_int;
                        double min_sig2_acc = ref_info.sig2_acc;
                        if (other_info.cur_track < ref_info.cur_track) {
                            min_sig2_int = other_info.sig2_int;
                            min_sig2_acc = other_info.sig2_acc;
                        }
                        
                        if (options_.overbounding_func != "none") {
                            sig2_int.block(row1, row2, n1, n2) = min_sig2_int * Eigen::MatrixXd::Identity(n1, n2);
                            sig2_int.block(row2, row1, n2, n1) = min_sig2_int * Eigen::MatrixXd::Identity(n2, n1);
                        }

                        if (options_.normal_func != "none") {
                            sig2_acc.block(row1, row2, n1, n2) = min_sig2_acc * Eigen::MatrixXd::Identity(n1, n2);
                            sig2_acc.block(row2, row1, n2, n1) = min_sig2_acc * Eigen::MatrixXd::Identity(n2, n1);
                        }
                    }
                }
            }
        }


        if (options_.use_complex_gnss_cov && gnss_measurement_pairs != nullptr && !gnss_measurement_pairs->empty()) {
            for (size_t i = 0; i < all_residuals.size(); ++i) {
                for (size_t j = i + 1; j < all_residuals.size(); ++j) {
                    const auto& ri = all_residuals[i];
                    const auto& rj = all_residuals[j];

                    if (ri.gnss_kind == GnssKind::None || rj.gnss_kind == GnssKind::None) {
                        continue;
                    }
                    if (ri.residual.size() != 1 || rj.residual.size() != 1) {
                        continue;
                    }

                    double cov = 0.0;
                    const bool same_time = std::fabs(ri.timestamp - rj.timestamp) < 1.0e-3;
                    const bool same_sat = (ri.prn == rj.prn);

                    const double rho_ij = computeGnssSpatialCorrelation(ri.azimuth, ri.elevation, rj.azimuth, rj.elevation);
                    const double rho_i_j_ref = computeGnssSpatialCorrelation(ri.azimuth, ri.elevation, rj.ref_azimuth, rj.ref_elevation);
                    const double rho_j_i_ref = computeGnssSpatialCorrelation(rj.azimuth, rj.elevation, ri.ref_azimuth, ri.ref_elevation);
                    const double rho_ref_ref = computeGnssSpatialCorrelation(ri.ref_azimuth, ri.ref_elevation, rj.ref_azimuth, rj.ref_elevation);

                    if (same_time &&
                        ri.gnss_kind == GnssKind::CodeDD &&
                        rj.gnss_kind == GnssKind::CodeDD) {
                        cov = 2.0 * (ri.sigma_user * rj.sigma_user * rho_ij + 
                                     ri.sigma_user_ref * rj.sigma_user_ref * rho_ref_ref -
                                     ri.sigma_user * rj.sigma_user_ref* rho_i_j_ref -
                                     rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref);
                    } else if (same_time &&
                               ri.gnss_kind == GnssKind::PhaseDD &&
                               rj.gnss_kind == GnssKind::PhaseDD) {
                        cov = (2.0 * (ri.sigma_user * rj.sigma_user * rho_ij + 
                                     ri.sigma_user_ref * rj.sigma_user_ref * rho_ref_ref -
                                     ri.sigma_user * rj.sigma_user_ref* rho_i_j_ref -
                                     rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref)) / (options_.user_Rr * options_.user_Rr);
                    } else if (!same_time && same_sat &&
                               ri.gnss_kind == GnssKind::CodeDD &&
                               rj.gnss_kind == GnssKind::CodeDD) {
                        const double dt = std::fabs(ri.timestamp - rj.timestamp);
                        const double decay = std::exp(-dt / std::max(1.0e-6, options_.tau_mp));
                        cov = 2.0 * options_.k_mp *
                              (ri.sigma_user * rj.sigma_user +
                               ri.sigma_user_ref * rj.sigma_user_ref -
                               ri.sigma_user * rj.sigma_user_ref * rho_i_j_ref -
                               rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref) *
                              decay;
                    } else if (!same_time && same_sat &&
                               ri.gnss_kind == GnssKind::PhaseDD &&
                               rj.gnss_kind == GnssKind::PhaseDD) {
                        const double dt = std::fabs(ri.timestamp - rj.timestamp);
                        const double decay = std::exp(-dt / std::max(1.0e-6, options_.tau_mp));
                        cov = (2.0 * options_.k_mp *
                               (ri.sigma_user * rj.sigma_user +
                                ri.sigma_user_ref * rj.sigma_user_ref -
                                ri.sigma_user * rj.sigma_user_ref * rho_i_j_ref -
                                rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref) *
                               decay) / (options_.user_Rr * options_.user_Rr);
                    // } else if (!same_time && !same_sat &&
                    //            ri.gnss_kind == GnssKind::CodeDD &&
                    //            rj.gnss_kind == GnssKind::CodeDD) {
                    //     const double dt = std::fabs(ri.timestamp - rj.timestamp);
                    //     const double decay = std::exp(-dt / std::max(1.0e-6, options_.tau_mp));
                    //     cov = 2.0 * options_.k_mp *
                    //           (ri.sigma_user * rj.sigma_user * rho_ij + 
                    //            ri.sigma_user_ref * rj.sigma_user_ref * rho_ref_ref -
                    //            ri.sigma_user * rj.sigma_user_ref* rho_i_j_ref -
                    //            rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref) *
                    //           decay;
                    // } else if (!same_time && !same_sat &&
                    //            ri.gnss_kind == GnssKind::PhaseDD &&
                    //            rj.gnss_kind == GnssKind::PhaseDD) {
                    //     const double dt = std::fabs(ri.timestamp - rj.timestamp);
                    //     const double decay = std::exp(-dt / std::max(1.0e-6, options_.tau_mp));
                    //     cov = (2.0 * options_.k_mp *
                    //            (ri.sigma_user * rj.sigma_user * rho_ij + 
                    //             ri.sigma_user_ref * rj.sigma_user_ref * rho_ref_ref -
                    //             ri.sigma_user * rj.sigma_user_ref* rho_i_j_ref -
                    //             rj.sigma_user * ri.sigma_user_ref * rho_j_i_ref) *
                    //            decay) / (options_.user_Rr * options_.user_Rr);
                    }


                    if (std::fabs(cov) > 0.0) {
                        const int ri_row = res_row_starts[i];
                        const int rj_row = res_row_starts[j];
                        sig2_int(ri_row, rj_row) = cov;
                        sig2_int(rj_row, ri_row) = cov;
                        sig2_acc(ri_row, rj_row) = cov;
                        sig2_acc(rj_row, ri_row) = cov;
                    }
                }
            }
        }

        sig2_int += 1.0e-10 * Eigen::MatrixXd::Identity(sig2_int.rows(), sig2_int.cols());
        sig2_acc += 1.0e-10 * Eigen::MatrixXd::Identity(sig2_acc.rows(), sig2_acc.cols());
    }

    CHECK_EQ(J_all.rows(), r_all.size());
    CHECK_EQ(J_all.rows(), sig2_int.rows());
    CHECK_EQ(J_all.rows(), row_ids_all.size());
    CHECK_EQ(J_all.cols(), col_ids_all.size());

    return true;
}

double VisualIntegrity::computeDualExpOverboundingSig2(std::vector<double> prm, int alpha, int beta) {

    // Parameters for dual-exponential model
    double a1,b1,a2,b2;
    if(prm.size() == 4){
        a1 = prm[0];
        b1 = prm[1];
        a2 = prm[2];
        b2 = prm[3];
    }else{
        LOG(ERROR) << "Error: The number of overbounding parameters is incorrect for dual exp!";
        return -1.0;
    }

    double sig_overbound = 0.0;
    for (size_t i = alpha + 1; i <= beta; i++){
        double sig = a1*exp(b1*i) + a2*exp(b2*i);
        sig_overbound = sig_overbound + sig*sig;
    }
    return sig_overbound;
}

void VisualIntegrity::extractLandmarkRowsCols(
    const FramePtr& frame,
    const PointMap& landmarks_map,
    const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
    const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
    const std::vector<std::pair<uint64_t, int>>& cols_curr,
    std::map<uint64_t, std::vector<int>>& curr_landmark_observation_rows,
    std::map<uint64_t, std::vector<int>>& all_landmark_observation_rows,
    std::map<uint64_t, int>& curr_landmark_object_ids,
    std::map<uint64_t, int>& all_landmark_object_ids)
{
    curr_landmark_observation_rows.clear();
    all_landmark_observation_rows.clear();
    curr_landmark_object_ids.clear();
    all_landmark_object_ids.clear();

    if (frame == nullptr || landmarks_map.empty()) {
        return;
    }

    std::unordered_map<uint64_t, std::vector<int>> resid_to_rows_map;
    std::unordered_map<uint64_t, std::string> resid_to_type_map;
    for (size_t i = 0; i < row_ids_all.size(); ++i) {
        resid_to_rows_map[row_ids_all[i].first].push_back(static_cast<int>(i));
        resid_to_type_map[row_ids_all[i].first] = row_ids_all[i].second;
    }

    std::unordered_map<uint64_t, std::vector<int>> lm_to_cols_all_map;
    for (size_t i = 0; i < col_ids_all.size(); ++i) {
        lm_to_cols_all_map[col_ids_all[i].first].push_back(static_cast<int>(i));
    }

    std::unordered_map<uint64_t, std::vector<int>> lm_to_cols_curr_map;
    for (const auto& pair : cols_curr) {
        lm_to_cols_curr_map[pair.first].push_back(pair.second);
    }

    const int latest_frame_id = frame->id();
    std::vector<BalancePoint> latest_frame_points;
    std::vector<size_t> latest_frame_feat_indices;
    std::set<size_t> latest_frame_seen_feats;
    for (const auto& lm_pair : landmarks_map) {
        const uint64_t lm_id = lm_pair.first.asInteger();
        if (lm_to_cols_all_map.find(lm_id) == lm_to_cols_all_map.end() &&
            lm_to_cols_curr_map.find(lm_id) == lm_to_cols_curr_map.end()) {
            continue;
        }
        for (const auto& obs_pair : lm_pair.second.observations) {
            if (obs_pair.first.frame_id != latest_frame_id) {
                continue;
            }
            const size_t feat_idx = obs_pair.first.keypoint_index_;
            if (!latest_frame_seen_feats.insert(feat_idx).second) {
                continue;
            }

            int obj_id = -1;
            if (feat_idx < frame->object_id_vec_.size()) {
                obj_id = frame->object_id_vec_[feat_idx];
            }
            latest_frame_points.push_back({frame->px_vec_(0, feat_idx), frame->px_vec_(1, feat_idx), obj_id});
            latest_frame_feat_indices.push_back(feat_idx);
        }
    }

    balanceObjectIds(latest_frame_points);

    std::map<size_t, int> latest_frame_balanced_ids;
    for (size_t i = 0; i < latest_frame_points.size(); ++i) {
        latest_frame_balanced_ids[latest_frame_feat_indices[i]] = latest_frame_points[i].id;
    }

    int random_id = 10000;
    for (const auto& lm_pair : landmarks_map) {
        const uint64_t lm_id = lm_pair.first.asInteger();
        const bool in_curr = lm_to_cols_curr_map.find(lm_id) != lm_to_cols_curr_map.end();
        const bool in_all = lm_to_cols_all_map.find(lm_id) != lm_to_cols_all_map.end();
        if (!in_curr && !in_all) {
            continue;
        }

        int curr_selected_object_id = -1;
        int all_selected_object_id = -1;
        int fallback_frame_id = std::numeric_limits<int>::min();

        for (const auto& obs_pair : lm_pair.second.observations) {
            const uint64_t residual_id = obs_pair.second;
            if (resid_to_type_map[residual_id] != "ReprojectionError") {
                // LOG(WARNING) << "Residual ID " << residual_id << " associated with landmark " << lm_id << " is not a ReprojectionError, the type is " << resid_to_type_map[residual_id];
                continue;
            }

            const auto row_it = resid_to_rows_map.find(residual_id);
            if (row_it != resid_to_rows_map.end()) {
                if (in_curr) {
                    auto& curr_rows = curr_landmark_observation_rows[lm_id];
                    curr_rows.insert(curr_rows.end(), row_it->second.begin(), row_it->second.end());
                }
                if (in_all) {
                    auto& all_rows = all_landmark_observation_rows[lm_id];
                    all_rows.insert(all_rows.end(), row_it->second.begin(), row_it->second.end());
                }
            }

            const int frame_id = obs_pair.first.frame_id;
            const size_t feat_idx = obs_pair.first.keypoint_index_;
            if (frame_id == latest_frame_id) {
                auto balanced_it = latest_frame_balanced_ids.find(feat_idx);
                if (balanced_it != latest_frame_balanced_ids.end() && balanced_it->second >= 0) {
                    if (in_curr) curr_selected_object_id = balanced_it->second;
                    if (in_all) all_selected_object_id = balanced_it->second;
                }
            }

            const auto frame_ptr = obs_pair.first.frame.lock();
            if (frame_ptr && feat_idx < frame_ptr->object_id_vec_.size()) {
                const int raw_object_id = frame_ptr->object_id_vec_[feat_idx];
                if (raw_object_id >= 0 && all_selected_object_id < 0 && frame_id > fallback_frame_id) {
                    fallback_frame_id = frame_id;
                    all_selected_object_id = raw_object_id;
                }
            }
        }

        if (in_all) {
            all_landmark_object_ids[lm_id] = all_selected_object_id;
        }

        if (in_curr) {
            if (curr_selected_object_id < 0 && all_selected_object_id >= 0) {
                curr_selected_object_id = all_selected_object_id;
            }
            curr_landmark_object_ids[lm_id] =
                (curr_selected_object_id < 0) ? random_id++ : curr_selected_object_id;
        }
    }

    for (auto& pair : curr_landmark_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    for (auto& pair : all_landmark_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
}

void VisualIntegrity::extractGnssRowsCols(
    const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
    const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
    const std::vector<std::pair<uint64_t, int>>& cols_curr,
    const std::map<uint64_t, std::vector<std::string>>& gnss_resid_to_prns,
    const std::map<uint64_t, std::vector<uint64_t>>& gnss_resid_to_param_ids,
    std::map<std::string, std::vector<int>>& curr_sat_observation_rows,
    std::map<std::string, std::vector<int>>& all_sat_observation_rows)
{
    curr_sat_observation_rows.clear();
    all_sat_observation_rows.clear();

    std::unordered_map<uint64_t, std::vector<int>> resid_to_rows_map;
    std::unordered_map<uint64_t, std::string> resid_to_type_map;
    for (size_t i = 0; i < row_ids_all.size(); ++i) {
        resid_to_rows_map[row_ids_all[i].first].push_back(static_cast<int>(i));
        resid_to_type_map[row_ids_all[i].first] = row_ids_all[i].second;
    }

    std::unordered_map<uint64_t, std::vector<int>> param_to_curr_cols_map;
    for (const auto& pair : cols_curr) {
        param_to_curr_cols_map[pair.first].push_back(pair.second);
    }

    auto is_supported_gnss_type = [](const std::string& type) {
        return type == "PseudorangeError" || type == "PseudorangeErrorSD" || type == "PseudorangeErrorDD" ||
               type == "PhaserangeError" || type == "PhaserangeErrorSD" || type == "PhaserangeErrorDD" ||
               type == "DopplerError";
    };

    std::set<std::string> curr_prns;
    for (const auto& prn_pair : gnss_resid_to_prns) {
        const uint64_t residual_id = prn_pair.first;
        const auto type_it = resid_to_type_map.find(residual_id);
        if (type_it == resid_to_type_map.end() || !is_supported_gnss_type(type_it->second)) {
            continue;
        }

        const auto params_it = gnss_resid_to_param_ids.find(residual_id);
        if (params_it == gnss_resid_to_param_ids.end()) {
            continue;
        }

        bool is_curr_residual = false;
        for (uint64_t param_id : params_it->second) {
            if (param_to_curr_cols_map.find(param_id) != param_to_curr_cols_map.end()) {
                is_curr_residual = true;
                break;
            }
        }
        if (!is_curr_residual) {
            continue;
        }

        for (const std::string& prn : prn_pair.second) {
            curr_prns.insert(prn);
        }
    }

    for (const auto& prn_pair : gnss_resid_to_prns) {
        const uint64_t residual_id = prn_pair.first;
        const auto rows_it = resid_to_rows_map.find(residual_id);
        if (rows_it == resid_to_rows_map.end()) {
            continue;
        }

        const auto type_it = resid_to_type_map.find(residual_id);
        if (type_it == resid_to_type_map.end() || !is_supported_gnss_type(type_it->second)) {
            if (!prn_pair.second.empty()) {
                // LOG(WARNING) << "Residual ID " << residual_id << " associated with GNSS satellite " << prn_pair.second.front() << " is not a supported GNSS error type, the type is " << type_it->second;
            }
            continue;
        }

        for (const std::string& prn : prn_pair.second) {
            auto& all_rows = all_sat_observation_rows[prn];
            all_rows.insert(all_rows.end(), rows_it->second.begin(), rows_it->second.end());

            if (curr_prns.count(prn) > 0) {
                auto& curr_rows = curr_sat_observation_rows[prn];
                curr_rows.insert(curr_rows.end(), rows_it->second.begin(), rows_it->second.end());
            }
        }
    }

    for (auto& pair : curr_sat_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    for (auto& pair : all_sat_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
}

void VisualIntegrity::extractImuRowsCols(
    const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
    const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
    const std::vector<std::pair<uint64_t, int>>& cols_curr,
    const std::map<uint64_t, std::vector<uint64_t>>& imu_resid_to_param_ids,
    std::map<uint64_t, std::vector<int>>& curr_imu_observation_rows,
    std::map<uint64_t, std::vector<int>>& all_imu_observation_rows)
{
    curr_imu_observation_rows.clear();
    all_imu_observation_rows.clear();

    std::unordered_map<uint64_t, std::vector<int>> resid_to_rows_map;
    std::unordered_map<uint64_t, std::string> resid_to_type_map;
    for (size_t i = 0; i < row_ids_all.size(); ++i) {
        resid_to_rows_map[row_ids_all[i].first].push_back(static_cast<int>(i));
        resid_to_type_map[row_ids_all[i].first] = row_ids_all[i].second;
    }

    std::unordered_map<uint64_t, std::vector<int>> param_to_curr_cols_map;
    for (const auto& pair : cols_curr) {
        param_to_curr_cols_map[pair.first].push_back(pair.second);
    }

    for (const auto& pair : imu_resid_to_param_ids) {
        const uint64_t residual_id = pair.first;
        const auto rows_it = resid_to_rows_map.find(residual_id);
        if (rows_it == resid_to_rows_map.end()) {
            continue;
        }
        if (resid_to_type_map[residual_id] != "IMUError") {
            // LOG(WARNING) << "Residual ID " << residual_id << " associated with IMU is not an IMUError, the type is " << resid_to_type_map[residual_id];
            continue;
        }

        auto& all_rows = all_imu_observation_rows[residual_id];
        all_rows.insert(all_rows.end(), rows_it->second.begin(), rows_it->second.end());

        bool connected_to_curr = false;
        for (uint64_t imu_state_id : pair.second) {
            if (param_to_curr_cols_map.find(imu_state_id) != param_to_curr_cols_map.end()) {
                connected_to_curr = true;
                break;
            }
        }

        if (connected_to_curr) {
            auto& curr_rows = curr_imu_observation_rows[residual_id];
            curr_rows.insert(curr_rows.end(), rows_it->second.begin(), rows_it->second.end());
        }
    }

    for (auto& pair : curr_imu_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    for (auto& pair : all_imu_observation_rows) {
        auto& rows = pair.second;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
}


void VisualIntegrity::extractPoseRelatedRowsCols(uint64_t current_pose_id,
                                                  IdType current_pose_type,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& pose_related_cols,
                                                  std::vector<int>& curr_pose_J_cols)
{


    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if ((pb_id.type() == current_pose_type) && pb_id.asInteger() == current_pose_id) {
            curr_pose_J_cols.push_back(cols_curr[i].second);
            pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
            if (curr_pose_J_cols.size() == 6) break;
            
        }
    }

    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if (pb_id.type() == IdType::ImuStates) {
            pb_id = changeIdType(pb_id, current_pose_type);
            if (pb_id.asInteger() == current_pose_id) {
                curr_pose_J_cols.push_back(cols_curr[i].second);
                pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
                if (curr_pose_J_cols.size() >= 9) break;
            }
        } 
    }

    if (curr_pose_J_cols.size() < 3) {
        for (size_t i = 0; i < cols_curr.size() && curr_pose_J_cols.size() < 3; ++i) {
            BackendId pb_id(cols_curr[i].first);
            if (pb_id.type() == current_pose_type && pb_id.asInteger() == current_pose_id) {
                curr_pose_J_cols.push_back(cols_curr[i].second);
            }
        }
    }
}

// --- MHSS Implementation ---

void VisualIntegrity::determineSubsets(const std::vector<double>& p_prior,
                                       std::vector<std::vector<int>>& subsets,
                                       std::vector<double>& pap_subset,
                                       double& p_not_monitored)
{
    int N = p_prior.size();
    double P_THRES = options_.P_THRES;
    
    if(p_prior.empty()){
        LOG(ERROR) << "Error: The prior probality of ISM is empty!";
    }
    
    //Deterimine the maximum simultanous faults need to monitor.
    std::vector<double> p_sum = p_prior;
    int N_fault_max = determineNfaultmax(p_sum, P_THRES);

    //Calculate the number of subsets.
    int N_used = N;
    int subsetsize = 0;
    for(int j = 0; j <= N_fault_max;++j){
        subsetsize = subsetsize + nchoosek((N_used),j);
    }
    LOG(INFO) << "The maximum simultanous faults need to monitor = " << N_fault_max << ", with measurement number = " << N << ", subset size = " << subsetsize << ", in P_THRES = " << P_THRES;

    //Initialize the subsets_ex and pap_subset.
    std::vector<std::vector<int>> subsets_ex(subsetsize);
    for (auto& col:subsets_ex){
        col.resize(N);
    }    
    std::fill(subsets_ex[0].begin(), subsets_ex[0].end(), 0);  //all-in-view (0 means no fault)
    pap_subset.resize(subsetsize);
    pap_subset[0] = 1;  


    //compute the probability of no fault occur
    double pnofault = 1.0; 
    for(int i = 0; i < N;++i){
        pnofault *= (1.0-p_prior[i]);
    }
    p_not_monitored = 1 - pnofault; 

    //Initialize k (number of simultaneous faults),p_not_monitored and subset index j   
    int k = 0;
    int j = 0;
    while ((k <= N_fault_max)&&(k <= N_used )&&(p_not_monitored > P_THRES)){
    
        //determine all the subsets of size k out of N_useds.
        std::vector<std::vector<int>> subsets_k_part = determine_k_subsets(N_used,k);

        //
        std::vector<std::vector<int>> subsets_k(subsets_k_part.size(), std::vector<int>(N, 1));
        std::vector<double> pap_subsets_k(subsets_k.size()); // evey row is the prior probability of fault mode k
        std::vector<double> p_diag((p_sum.size()));
        for(int i = 0; i < p_sum.size(); ++i){
            p_diag[i] = p_sum[i] / (1 - p_sum[i]);
            if(p_sum[i] == 0) p_diag[i] = 1.0;
        }
        for(int i = 0; i < subsets_k_part.size(); ++i)
        {
            double product = 1.0;
            int h_Col = 0;
            for(int jj = 0; jj < p_sum.size(); ++jj)
            {
                if(p_sum[jj] != 0 && h_Col < subsets_k_part[0].size())
                {
                    subsets_k[i][jj] = subsets_k_part[i][h_Col]; 
                    ++h_Col;
                }
                if(subsets_k[i][jj]){
                    product *= p_diag[jj];
                }
            }
            product *= pnofault;
            pap_subsets_k[i] = product;
        }

        //sort subsets by decreasing probability
        std::vector<size_t> index(pap_subsets_k.size());
        std::iota(index.begin(),index.end(),0);
        std::sort(index.begin(),index.end(),[&](size_t i1, size_t i2){
            return pap_subsets_k[i1] > pap_subsets_k[i2];
        });
        std::vector<double> p_subsets_k_s(pap_subsets_k.size());
        for(int i = 0; i < pap_subsets_k.size(); ++i){
            p_subsets_k_s[i] = pap_subsets_k[index[i]];
        }
        std::vector<std::vector<int>> subsets_k_s(subsets_k.size());
        for(int i = 0; i < subsets_k.size(); ++i){
            subsets_k_s[i] = subsets_k[index[i]];
        }

        //k->all
        int h = 0;
        while ((h < subsets_k_s.size())&&(p_not_monitored > P_THRES))
        {     
            if(p_subsets_k_s[h] > 0)
            {        
                subsets_ex[j] = subsets_k_s[h];
                pap_subset[j] = p_subsets_k_s[h]; 
                if( k !=0 ) p_not_monitored = p_not_monitored - pap_subset[j];
                ++j;
            }
            ++h;
        }
        ++k;
    }

    pap_subset.resize(j);    
    subsets_ex.resize(j);

    subsets = subsets_ex;
    
    // Flip bits: 0 (fault) -> 0 (excluded), 1 (no fault) -> 1 (used)
    // Wait, in integrity.cpp:
    // subsets_ex[0] is all 0s (all-in-view).
    // subsets_k[i][jj] = 1 means fault.
    // So 0 means used, 1 means fault.
    // Then at the end: i = 1 - i.
    // So 1 becomes 0 (excluded), 0 becomes 1 (used).
    
    for ( auto& col : subsets){
        for (auto& i : col){
            i = 1 - i;
        }
    }
}

void VisualIntegrity::computeSubsetSolution(const Eigen::MatrixXd& J,
                                            const Eigen::VectorXd& residual,
                                            const Eigen::MatrixXd& sig2_int,
                                            const Eigen::MatrixXd& sig2_acc,
                                            const std::vector<std::vector<int>>& subsets,
                                            const std::vector<std::vector<int>>& fault_group_rows,
                                            const std::vector<int>& curr_pose_J_cols,
                                            Eigen::MatrixXd& sigma_out,  
                                            Eigen::MatrixXd& bias_out,
                                            Eigen::MatrixXd& sigma_ss_out,
                                            Eigen::MatrixXd& bias_ss_out,
                                            Eigen::MatrixXd& s1vec_out,
                                            Eigen::MatrixXd& s2vec_out,
                                            Eigen::MatrixXd& s3vec_out,
                                            Eigen::MatrixXd& x_out,
                                            Eigen::VectorXd& chi2)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int N_sets = subsets.size();
    int N_J_rows = J.rows();
    int N_J_cols = J.cols();
    int N_state = 3;
    int N_meas_curr = subsets[0].size(); // Number of fault groups

    // Initialize outputs
    using MatrixRowMaj    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    MatrixRowMaj sigma    = MatrixRowMaj::Constant(N_sets, 3, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj bias     = MatrixRowMaj::Constant(N_sets, 3, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj sigma_ss = MatrixRowMaj::Constant(N_sets, 3, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj bias_ss  = MatrixRowMaj::Constant(N_sets, 3, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj s1vec    = MatrixRowMaj::Constant(N_sets, N_J_rows, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj s2vec    = MatrixRowMaj::Constant(N_sets, N_J_rows, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj s3vec    = MatrixRowMaj::Constant(N_sets, N_J_rows, std::numeric_limits<double>::quiet_NaN());
    MatrixRowMaj x        = MatrixRowMaj::Zero(N_sets, 3);
    chi2 = Eigen::VectorXd::Zero(N_sets);
    Eigen::VectorXd nom_bias = Eigen::VectorXd::Zero(N_J_rows);

    // Robust weight matrix computation with validation
    Eigen::MatrixXd W, W_acc;
    bool diag_force = false;
    Eigen::MatrixXd sig2_int_copy = sig2_int; //convert const
    Eigen::MatrixXd sig2_acc_copy = sig2_acc;
    LOG(INFO) << "Compute the inverse of sig2_int:";
    bool weight_computed = computeRobustWeightMatrix(sig2_int_copy, W, diag_force);
    LOG(INFO) << "Compute the inverse of sig2_acc:";
    bool weight_acc_computed = computeRobustWeightMatrix(sig2_acc_copy, W_acc, diag_force);
    if (!weight_computed) {
        LOG(ERROR) << "Failed to compute valid weight matrix";
        return;
    }
    const bool W_is_diagonal = W.isDiagonal(0.0);
    bool is_sig2_int_diag = sig2_int_copy.isDiagonal(1e-6);
    bool is_sig2_acc_diag = sig2_acc_copy.isDiagonal(1e-6);
    Eigen::VectorXd sig2_int_diag = is_sig2_int_diag ? sig2_int_copy.diagonal() : Eigen::VectorXd();
    Eigen::VectorXd sig2_acc_diag = is_sig2_acc_diag ? sig2_acc_copy.diagonal() : Eigen::VectorXd();
    
    // Precompute All-in-view Cholesky Decomposition
    // Precompute J'W (N_cols x N_J_rows)
    Eigen::MatrixXd JtW_all = J.transpose() * W;
    // Precompute b_all = J'W * r
    Eigen::VectorXd b_all = JtW_all * residual;
    // Precompute J'WJ
    Eigen::MatrixXd JtWJ_all = JtW_all * J;

    // Robust Cholesky decomposition with fallback mechanisms
    Eigen::LLT<Eigen::MatrixXd> llt_all;
    double used_damping = 0.0;
    LOG(INFO) << "Compute the Cholesky decomposition of JtWJ:";
    bool cholesky_success = computeRobustCholesky(JtWJ_all, llt_all, used_damping);
    if (!cholesky_success) {
        LOG(ERROR) << "All Cholesky decomposition attempts failed!";
        return;
    }
    Eigen::MatrixXd P_all = llt_all.solve(Eigen::MatrixXd::Identity(N_J_cols, N_J_cols));

    // all-in-view solution
    Eigen::VectorXd x_all = llt_all.solve(b_all);
    std::vector<Eigen::VectorXd> s_base(3);
    for (int k = 0; k < 3; ++k) {
        x(0, k) = x_all(curr_pose_J_cols[k]);
        Eigen::VectorXd e_k = Eigen::VectorXd::Zero(N_J_cols);
        e_k(curr_pose_J_cols[k]) = 1.0;
        Eigen::VectorXd p_col = llt_all.solve(e_k);
        s_base[k] = p_col.transpose() * JtW_all;
        if (k == 0) s1vec.row(0) = s_base[k];
        if (k == 1) s2vec.row(0) = s_base[k];
        if (k == 2) s3vec.row(0) = s_base[k];
        double var = is_sig2_int_diag ? 
                     s_base[k].cwiseAbs2().dot(sig2_int_diag) : 
                     (s_base[k].transpose() * sig2_int_copy * s_base[k]).value();
        sigma(0, k) = std::sqrt(var);
        bias(0, k) = s_base[k].cwiseAbs().dot(nom_bias);
        sigma_ss(0, k) = 0.0;
        bias_ss(0, k) = 0.0;
    }

    LOG(INFO) << "Computing subset solutions for " << N_sets - 1 << " subsets.";
    
    // Progress tracking for subset computation
    std::atomic<int> subsets_processed{0};
    std::atomic<int> last_progress{0};
    std::vector<std::vector<int>> group_rows_cache(N_meas_curr);

    // Pre-cache rows for each fault group.
    for (int j = 0; j < N_meas_curr; ++j) {
        if (j < static_cast<int>(fault_group_rows.size())) {
            group_rows_cache[j] = fault_group_rows[j];
            std::sort(group_rows_cache[j].begin(), group_rows_cache[j].end());
            group_rows_cache[j].erase(std::unique(group_rows_cache[j].begin(), group_rows_cache[j].end()), group_rows_cache[j].end());
        }
    }
    
    // Pre-allocate thread-local storage to avoid dynamic allocation in parallel loop
    #pragma omp parallel
    {
        std::vector<int> rows_to_remove;
        rows_to_remove.reserve(N_meas_curr * 2);
        size_t n_max = 100;
        Eigen::MatrixXd J_rem(n_max, N_J_cols);
        Eigen::MatrixXd W_rem(n_max, n_max);
        Eigen::VectorXd r_rem(n_max);
        
        // Hoist allocations to thread-local scope to avoid repeated mallocs
        Eigen::MatrixXd JP; 
        Eigen::MatrixXd W_rem_inv;
        Eigen::MatrixXd Middle; 
        Eigen::MatrixXd Kernel;
        Eigen::MatrixXd UpdateBlock;
        Eigen::VectorXd b_sub; 
        Eigen::VectorXd x_curr_full;
        Eigen::RowVectorXd P_sub_row;           
        Eigen::VectorXd s_row;  
        Eigen::VectorXd ds;

        #pragma omp for schedule(dynamic, 16)  // Dynamic scheduling for load balancing
        for (int i = 1; i < N_sets; ++i) {
            rows_to_remove.clear();
            int groups_to_remove = 0;
            // Identify rows to remove based on current subset
            for (int j = 0; j < N_meas_curr; ++j) {
                if (subsets[i][j] == 0) { // 0 means fault/exclude
                    ++groups_to_remove;
                    const auto& rows = group_rows_cache[j];
                    rows_to_remove.insert(rows_to_remove.end(), rows.begin(), rows.end());
                    // LOG(INFO) << "Subset " << i << ": Excluding fault group " << j << " with " << rows.size() << " rows.";
                }
            }

            std::sort(rows_to_remove.begin(), rows_to_remove.end());
            rows_to_remove.erase(std::unique(rows_to_remove.begin(), rows_to_remove.end()), rows_to_remove.end());

            // Check observability roughly
            const int remaining_rows = N_J_rows - static_cast<int>(rows_to_remove.size());
            if (!rows_to_remove.empty() && remaining_rows < 6) {
                LOG(WARNING) << "Subset " << i << " skipped due to insufficient measurements for observability.";
                continue;
            }


            // Build J_rem and W_rem efficiently
            size_t n_rem = rows_to_remove.size();
            J_rem.resize(n_rem, N_J_cols);
            W_rem.resize(n_rem, n_rem);
            r_rem.resize(n_rem);

            for(size_t r = 0; r < n_rem; ++r) {
                int r_idx = rows_to_remove[r];
                J_rem.row(r) = J.row(r_idx);
                r_rem(r) = residual(r_idx);
                W_rem(r, r) = W(r_idx, r_idx);
                if (!W_is_diagonal) {
                    for(size_t c = r + 1; c < n_rem; ++c) {
                        int c_idx = rows_to_remove[c];
                        W_rem(r, c) = W(r_idx, c_idx);
                        W_rem(c, r) = W(c_idx, r_idx); 
                    }
                }
            }

            // UpdateBlock = (P_all * J_rem^T) * (W_rem^-1 - J_rem * P_all * J_rem^T)^-1
            JP = J_rem * P_all; // n_rem x N_cols
            if (W_is_diagonal) {
                W_rem_inv = W_rem.diagonal().cwiseInverse().asDiagonal();
            } else {
                W_rem_inv = robustInverse(W_rem);
            }
            
            Middle = W_rem_inv - (JP * J_rem.transpose());
            Kernel = robustInverse(Middle);
            UpdateBlock = JP.transpose() * Kernel;
            b_sub = b_all - J_rem.transpose() * W_rem * r_rem;
            x_curr_full = P_all * b_sub + UpdateBlock * (JP * b_sub);

            // Compute S vectors and Sigma for all 3 dimensions
            for (int k = 0; k < 3; ++k) {
                int row_id = curr_pose_J_cols[k];
                x(i, k) = x_curr_full(row_id);
                // Compute specific row of P_sub: P_row + UpdateBlock_row * JP
                P_sub_row = P_all.row(row_id) + UpdateBlock.row(row_id) * JP;
                // Compute S vector: S = P_sub_row * JtW_all
                s_row = (P_sub_row * JtW_all).transpose();
                // Set S values for removed measurements to 0
                for(size_t r=0; r<rows_to_remove.size(); ++r) {
                    s_row(rows_to_remove[r]) = 0.0;
                }

                // Store S vectors
                if(k==0) s1vec.row(i) = s_row;
                if(k==1) s2vec.row(i) = s_row;
                if(k==2) s3vec.row(i) = s_row;
                ds = s_row - s_base[k];
                
                // Compute Sigma and Sigma_ss with diagonal optimization
                double var = is_sig2_int_diag ? 
                         s_row.cwiseAbs2().dot(sig2_int_diag) : 
                         (s_row.transpose() * sig2_int_copy * s_row).value();
                double var_ss = is_sig2_acc_diag ? 
                         ds.cwiseAbs2().dot(sig2_acc_diag) : 
                         (ds.transpose() * sig2_acc_copy * ds).value();
                
                sigma(i, k) = std::sqrt(var);
                bias(i, k) = s_row.cwiseAbs().dot(nom_bias);
                sigma_ss(i, k) = std::sqrt(var_ss);
                bias_ss(i, k) = ds.cwiseAbs().dot(nom_bias);
                
                if (std::isnan(sigma(i, k)) || std::isnan(sigma_ss(i, k))) {
            
                    // Debug: Check for NaN in s_row before computing variances
                    bool s_row_has_nan = false;
                    bool s_row_has_inf = false;
                    for (int idx = 0; idx < s_row.size(); ++idx) {
                        if (std::isnan(s_row(idx))) {
                            s_row_has_nan = true;
                            LOG(WARNING) << "NaN found in s_row at index " << idx << " for subset " << i << ", dimension " << k;
                        }
                        if (std::isinf(s_row(idx))) {
                            s_row_has_inf = true;
                            LOG(WARNING) << "Inf found in s_row at index " << idx << " for subset " << i << ", dimension " << k;
                        }
                    }
                    
                    // Debug: Check for NaN in ds
                    bool ds_has_nan = false;
                    bool ds_has_inf = false;
                    for (int idx = 0; idx < ds.size(); ++idx) {
                        if (std::isnan(ds(idx))) {
                            ds_has_nan = true;
                            LOG(WARNING) << "NaN found in ds at index " << idx << " for subset " << i << ", dimension " << k;
                        }
                        if (std::isinf(ds(idx))) {
                            ds_has_inf = true;
                            LOG(WARNING) << "Inf found in ds at index " << idx << " for subset " << i << ", dimension " << k;
                        }
                    }
                    
                    // Debug: Check sig2_int_copy and sig2_acc_copy for issues
                    bool sig2_int_has_issue = false;
                    bool sig2_acc_has_issue = false;
                    if (!is_sig2_int_diag) {
                        for (int idx = 0; idx < sig2_int_copy.rows(); ++idx) {
                            if (std::isnan(sig2_int_copy(idx, idx)) || std::isinf(sig2_int_copy(idx, idx)) || sig2_int_copy(idx, idx) <= 0.0) {
                                sig2_int_has_issue = true;
                                LOG(WARNING) << "sig2_int_copy has issue at diagonal index " << idx << ": " << sig2_int_copy(idx, idx);
                            }
                        }
                    } else {
                        for (int idx = 0; idx < sig2_int_diag.size(); ++idx) {
                            if (std::isnan(sig2_int_diag(idx)) || std::isinf(sig2_int_diag(idx)) || sig2_int_diag(idx) <= 0.0) {
                                sig2_int_has_issue = true;
                                LOG(WARNING) << "sig2_int_diag has issue at index " << idx << ": " << sig2_int_diag(idx);
                            }
                        }
                    }
                    
                    if (!is_sig2_acc_diag) {
                        for (int idx = 0; idx < sig2_acc_copy.rows(); ++idx) {
                            if (std::isnan(sig2_acc_copy(idx, idx)) || std::isinf(sig2_acc_copy(idx, idx)) || sig2_acc_copy(idx, idx) <= 0.0) {
                                sig2_acc_has_issue = true;
                                LOG(WARNING) << "sig2_acc_copy has issue at diagonal index " << idx << ": " << sig2_acc_copy(idx, idx);
                            }
                        }
                    } else {
                        for (int idx = 0; idx < sig2_acc_diag.size(); ++idx) {
                            if (std::isnan(sig2_acc_diag(idx)) || std::isinf(sig2_acc_diag(idx)) || sig2_acc_diag(idx) <= 0.0) {
                                sig2_acc_has_issue = true;
                                LOG(WARNING) << "sig2_acc_diag has issue at index " << idx << ": " << sig2_acc_diag(idx);
                            }
                        }
                    }
                
                    // Debug: Check intermediate computation results
                    if (std::isnan(var) || std::isinf(var)) {
                        LOG(WARNING) << "var is NaN/Inf for subset " << i << ", dimension " << k;
                        LOG(INFO) << "var computation details: is_sig2_int_diag=" << is_sig2_int_diag 
                                << ", s_row.norm()=" << s_row.norm() 
                                << ", sig2_int_diag.norm()=" << (is_sig2_int_diag ? sig2_int_diag.norm() : -1.0);
                        if (!is_sig2_int_diag) {
                            LOG(INFO) << "sig2_int_copy matrix norm: " << sig2_int_copy.norm();
                        }
                    }
                    
                    if (std::isnan(var_ss) || std::isinf(var_ss)) {
                        LOG(WARNING) << "var_ss is NaN/Inf for subset " << i << ", dimension " << k;
                        LOG(INFO) << "var_ss computation details: is_sig2_acc_diag=" << is_sig2_acc_diag 
                                << ", ds.norm()=" << ds.norm() 
                                << ", sig2_acc_diag.norm()=" << (is_sig2_acc_diag ? sig2_acc_diag.norm() : -1.0);
                        if (!is_sig2_acc_diag) {
                            LOG(INFO) << "sig2_acc_copy matrix norm: " << sig2_acc_copy.norm();
                        }
                    }
                    LOG(WARNING) << "Warning: NaN encountered in sigma computation for subset " << i << ", dimension " << k;
                    LOG(WARNING) << "Debug Info: var = " << var << ", var_ss = " << var_ss;
                    LOG(WARNING) << "Debug Info: s_row norm = " << s_row.norm() << ", ds norm = " << ds.norm();
                    LOG(WARNING) << "Debug Info: s_row has NaN = " << s_row_has_nan << ", s_row has Inf = " << s_row_has_inf;
                    LOG(WARNING) << "Debug Info: ds has NaN = " << ds_has_nan << ", ds has Inf = " << ds_has_inf;
                    LOG(WARNING) << "Debug Info: sig2_int has issue = " << sig2_int_has_issue << ", sig2_acc has issue = " << sig2_acc_has_issue;
                    LOG(WARNING) << "Debug Info: subset index = " << i << ", dimension = " << k << ", groups_to_remove = " << groups_to_remove;
                    LOG(WARNING) << "Debug Info: rows_to_remove size = " << rows_to_remove.size() << ", N_meas_curr = " << N_meas_curr;
                    LOG(WARNING) << "Debug Info: P_all norm = " << P_all.norm() << ", JtW_all norm = " << JtW_all.norm();
                    LOG(WARNING) << "Debug Info: JP norm = " << JP.norm() << ", W_rem_inv norm = " << W_rem_inv.norm();
                    LOG(WARNING) << "Debug Info: Middle norm = " << Middle.norm() << ", Kernel norm = " << Kernel.norm();
                    LOG(WARNING) << "Debug Info: UpdateBlock norm = " << UpdateBlock.norm() << ", b_sub norm = " << b_sub.norm();
                    LOG(WARNING) << "Debug Info: x_curr_full norm = " << x_curr_full.norm() << ", P_sub_row norm = " << P_sub_row.norm();
                    // saveEigenMatrixToFile(sig2_int, "/home/syl/GICI-IM/results/debug/sig2_int_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    // saveEigenMatrixToFile(sig2_acc, "/home/syl/GICI-IM/results/debug/sig2_acc_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    // saveEigenMatrixToFile(s1vec_out, "/home/syl/GICI-IM/results/debug/s1vec_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    // saveEigenMatrixToFile(s2vec_out, "/home/syl/GICI-IM/results/debug/s2vec_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    // saveEigenMatrixToFile(s3vec_out, "/home/syl/GICI-IM/results/debug/s3vec_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    saveEigenMatrixToFile(sigma_out, "/home/syl/GICI-IM/results/debug/sigma_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    saveEigenMatrixToFile(sigma_ss_out, "/home/syl/GICI-IM/results/debug/sigma_ss_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                }
            }  
            // Update progress
            int current_count = subsets_processed.fetch_add(1) + 1;
            int progress = static_cast<int>((static_cast<double>(current_count)/ (N_sets - 1)) * 100);
            int last_prog = last_progress.load();
            if (progress != last_prog && progress % 20 == 0 && last_progress.compare_exchange_strong(last_prog, progress)) {
                LOG(INFO) << "Subset computation progress: " << progress << "% (" << current_count << "/" << N_sets - 1 << ")";
            }
        }
        
    }
    // about time taken
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(INFO) << "Compute subset solution, number of subsets: " << N_sets << ", measurement dimension: " << sig2_int.rows() << ", time taken(s): " << elapsed.count();
    
    // Cast back to standard MatrixXd if necessary (Eigen handles implicit copy)
    sigma_out = sigma;
    bias_out = bias;
    sigma_ss_out = sigma_ss;
    bias_ss_out = bias_ss;
    s1vec_out = s1vec;
    s2vec_out = s2vec;
    s3vec_out = s3vec;
    x_out = x;
}

// Helper function to compute robust weight matrix with validation
// sig2 is both input and output: it will be modified to be positive definite if needed
// Helper function to compute robust weight matrix with validation
// sig2 is both input and output: it will be modified to be positive definite if needed
bool VisualIntegrity::computeRobustWeightMatrix(Eigen::MatrixXd& sig2, Eigen::MatrixXd& W, bool& diag_force) {
    if (sig2.rows() == 0 || sig2.cols() == 0) {
        LOG(ERROR) << "   - Empty sig2 matrix";
        return false;
    }
    
    // Constants for regularization
    const double REGULARIZATION_EPS = 1e-8;
    const int MAX_REGULARIZATION_ATTEMPTS = 5;
    
    // Check if sig2 is diagonal (within numerical tolerance)
    bool is_diagonal = sig2.isDiagonal(1e-12);
    
    // Handle diagonal matrices efficiently
    if (is_diagonal) {
        // Extract diagonal elements
        Eigen::VectorXd diag_elements = sig2.diagonal();
        
        // Validate and regularize diagonal elements
        bool modified = false;
        for (int i = 0; i < diag_elements.size(); ++i) {
            if (diag_elements(i) <= 0.0 || !std::isfinite(diag_elements(i))) {
                LOG(WARNING) << "   - Invalid variance at index " << i << ": " << diag_elements(i) 
                           << ", using regularization";
                LOG(ERROR) << "   - Something wrong in sig2_int/sig2_acc, maybe in saveSnap process!";
                diag_elements(i) = 1e-6; // Apply regularization
                modified = true;
            }
        }
        
        // Update sig2 if any modifications were made
        if (modified) {
            sig2 = diag_elements.asDiagonal();
        }
        
        // Create diagonal weight matrix (inverse of variances)
        W = diag_elements.cwiseInverse().asDiagonal();
        LOG(INFO) << "   - Using diagonal weight matrix from sig2";
        return true;
    }
    
    // For non-diagonal matrices, attempt to preserve the original structure
    // First, save the original matrix for reference
    Eigen::MatrixXd sig2_original = sig2;

    if (diag_force) {
        // Extract diagonal elements from original matrix
        Eigen::VectorXd diag_elements = sig2_original.diagonal();
        
        // Calculate trace for preservation
        double trace_original = sig2_original.trace();
        double trace_diag = diag_elements.sum();
        
        // Scale diagonal elements to preserve total variance (if trace is meaningful)
        if (trace_diag > 1e-12 && trace_original > 1e-12) {
            double relative_diff = std::abs(trace_original - trace_diag) / trace_original;
            if (relative_diff > 0.1) { // More than 10% difference
                double scale_factor = trace_original / trace_diag;
                diag_elements *= scale_factor;
                LOG(WARNING) << "   - Scaling diagonal elements by factor " << scale_factor 
                        << " to preserve trace (relative diff: " << relative_diff << ")";
            }
        }
        
        // Validate and regularize diagonal elements
        for (int i = 0; i < diag_elements.size(); ++i) {
            if (diag_elements(i) <= 0.0 || !std::isfinite(diag_elements(i))) {
                // Use original diagonal value if available, otherwise use default
                double original_val = sig2_original(i, i);
                diag_elements(i) = (std::isfinite(original_val) && original_val > 0) ? 
                                std::max(1e-6, original_val) : 1e-6;
                LOG(WARNING) << "   - Regularizing diagonal element " << i 
                            << " to " << diag_elements(i);
            }
        }
        
        // Create final diagonal matrices
        sig2 = diag_elements.asDiagonal();
        W = diag_elements.cwiseInverse().asDiagonal();
        
        LOG(WARNING) << "   - Because sig2_int uses diag, forcing sig2_acc to diagonal matrix with trace";
        return true;
    }
    
    // Step 1: Ensure symmetry (numerical stability)
    Eigen::MatrixXd sig2_sym = (sig2 + sig2.transpose()) / 2.0;
    
    // Step 2: Validate diagonal elements
    bool needs_regularization = false;
    for (int i = 0; i < sig2_sym.rows(); ++i) {
        double diag_val = sig2_sym(i, i);
        if (diag_val <= 0.0 || !std::isfinite(diag_val)) {
            LOG(WARNING) << "   - Invalid variance at index " << i << ": " << diag_val;
            needs_regularization = true;
            break;
        }
    }
    
    // Step 3: Check positive definiteness using Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt_test(sig2_sym);
    if (llt_test.info() == Eigen::Success && !needs_regularization) {
        // Matrix is positive definite, attempt to compute inverse
        try {
            W = robustInverse(sig2_sym);
            sig2 = sig2_sym; // Update to symmetric version
            LOG(INFO) << "   - Successfully inverted positive definite matrix";
            return true;
        } catch (...) {
            LOG(WARNING) << "   - Failed to invert positive definite matrix";
            needs_regularization = true;
        }
    } else {
        LOG(WARNING) << "   - sig2 is not positive definite, attempting regularization";
        needs_regularization = true;
    }
    
    // Step 4: Attempt incremental regularization before falling back to diagonal
    if (needs_regularization) {
        bool regularization_success = false;
        
        // Try multiple regularization levels
        for (int attempt = 0; attempt < MAX_REGULARIZATION_ATTEMPTS && !regularization_success; ++attempt) {
            double epsilon = REGULARIZATION_EPS * std::pow(10.0, attempt);
            Eigen::MatrixXd regularized = sig2_sym;
            
            // Add regularization to diagonal
            for (int i = 0; i < regularized.rows(); ++i) {
                regularized(i, i) += epsilon;
            }
            
            // Ensure symmetry after regularization
            regularized = (regularized + regularized.transpose()) / 2.0;
            
            // Test positive definiteness
            Eigen::LLT<Eigen::MatrixXd> llt_reg(regularized);
            if (llt_reg.info() == Eigen::Success) {
                try {
                    W = robustInverse(regularized);
                    sig2 = regularized;
                    regularization_success = true;
                    LOG(WARNING) << "   - Regularization succeeded with epsilon = " << epsilon;
                    break;
                } catch (...) {
                    // Continue to next attempt with larger epsilon
                }
            }
        }
        
        if (regularization_success) {
            return true;
        }
    }
    
    // Step 5: Attempt to find nearest positive definite matrix using eigenvalue correction
    LOG(WARNING) << "   - Regularization failed, attempting nearest positive definite approximation";
    
    try {
        // Ensure symmetry
        Eigen::MatrixXd H = (sig2_sym + sig2_sym.transpose()) / 2.0;
        
        // Compute eigenvalue decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }
        
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
        
        // Find minimum eigenvalue and maximum absolute eigenvalue
        double min_eigenvalue = eigenvalues.minCoeff();
        double max_abs_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
        
        // Correct negative or near-zero eigenvalues
        bool eigenvalues_corrected = false;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (eigenvalues(i) < REGULARIZATION_EPS) {
                eigenvalues(i) = REGULARIZATION_EPS;
                eigenvalues_corrected = true;
            }
        }
        
        if (eigenvalues_corrected) {
            LOG(WARNING) << "   - Corrected " << eigenvalues_corrected 
                        << " eigenvalues, min eigenvalue was: " << min_eigenvalue;
        }
        
        // Reconstruct the nearest positive definite matrix
        Eigen::MatrixXd nearest_pd = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        
        // Ensure final symmetry
        nearest_pd = (nearest_pd + nearest_pd.transpose()) / 2.0;
        
        // Validate the reconstructed matrix
        Eigen::LLT<Eigen::MatrixXd> llt_final(nearest_pd);
        if (llt_final.info() == Eigen::Success) {
            try {
                W = robustInverse(nearest_pd);
                sig2 = nearest_pd;
                LOG(WARNING) << "   - Successfully inverted positive definite matrix using nearest positive definite matrix approximation";
                return true;
            } catch (...) {
                LOG(WARNING) << "   - Failed to invert nearest PD matrix";
            }
        }
    } catch (const std::exception& e) {
        LOG(WARNING) << "   - Nearest PD computation failed: " << e.what();
    } catch (...) {
        LOG(WARNING) << "   - Unknown error in nearest PD computation";
    }
    
    // Step 6: Final fallback - diagonal approximation with trace preservation
    LOG(WARNING) << "   - All methods failed, falling back to diagonal approximation";
    
    // Extract diagonal elements from original matrix
    Eigen::VectorXd diag_elements = sig2_original.diagonal();
    
    // Calculate trace for preservation
    double trace_original = sig2_original.trace();
    double trace_diag = diag_elements.sum();
    
    // Scale diagonal elements to preserve total variance (if trace is meaningful)
    if (trace_diag > 1e-12 && trace_original > 1e-12) {
        double relative_diff = std::abs(trace_original - trace_diag) / trace_original;
        if (relative_diff > 0.1) { // More than 10% difference
            double scale_factor = trace_original / trace_diag;
            diag_elements *= scale_factor;
            LOG(WARNING) << "   - Scaling diagonal elements by factor " << scale_factor 
                     << " to preserve trace (relative diff: " << relative_diff << ")";
        }
    }
    
    // Validate and regularize diagonal elements
    for (int i = 0; i < diag_elements.size(); ++i) {
        if (diag_elements(i) <= 0.0 || !std::isfinite(diag_elements(i))) {
            // Use original diagonal value if available, otherwise use default
            double original_val = sig2_original(i, i);
            diag_elements(i) = (std::isfinite(original_val) && original_val > 0) ? 
                              std::max(1e-6, original_val) : 1e-6;
            LOG(WARNING) << "   - Regularizing diagonal element " << i 
                        << " to " << diag_elements(i);
        }
    }
    
    // Create final diagonal matrices
    sig2 = diag_elements.asDiagonal();
    W = diag_elements.cwiseInverse().asDiagonal();
    
    LOG(WARNING) << "   - Fallback to diagonal matrix with trace: " << sig2.trace() << ", and will force sig2_acc to diag";
    diag_force = true;
    
    return true;
}


Eigen::MatrixXd VisualIntegrity::robustInverse(const Eigen::MatrixXd& M, double svd_threshold, bool always_pseudo) {
    
    if (M.rows() == 0 || M.cols() == 0) {
        return Eigen::MatrixXd::Zero(0, 0);
    }
    
    // Option 1: Always use pseudoinverse (most robust)
    if (always_pseudo) {
        return pseudoinverseSVD(M, svd_threshold);
    }
    
    // Option 2: For square matrices, try regular inverse first
    if (M.rows() == M.cols()) {
        // Fast path: diagonal matrix
        if (M.isDiagonal(1e-12)) {
            return M.diagonal().cwiseInverse().asDiagonal();
        }
        
        // Try direct inverse
        try {
            Eigen::MatrixXd inv = M.inverse();
            
            // Verify accuracy
            Eigen::MatrixXd I_check = M * inv;
            double error = (I_check - Eigen::MatrixXd::Identity(M.rows(), M.rows())).norm();
            
            if (error < M.rows() * 1e-12 * M.norm()) {
                return inv;
            }
        } catch (...) {
            LOG(WARNING) << "   - Direct inverse failed, trying alternatives.";
        }
        
        // Try Cholesky for symmetric positive definite
        if (M.isApprox(M.transpose(), 1e-12)) {
            Eigen::LLT<Eigen::MatrixXd> llt(M);
            if (llt.info() == Eigen::Success) {
                return llt.solve(Eigen::MatrixXd::Identity(M.rows(), M.rows()));
            }
        }
    }
    
    // Option 3: Use pseudoinverse (handles all cases)
    return pseudoinverseSVD(M, svd_threshold);
}


Eigen::MatrixXd VisualIntegrity::pseudoinverseSVD(const Eigen::MatrixXd& M, double threshold) {
    
    if (M.norm() < std::numeric_limits<double>::min()) {
        return Eigen::MatrixXd::Zero(M.cols(), M.rows());
    }
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sv = svd.singularValues();
    
    if (sv.size() == 0) {
        return Eigen::MatrixXd::Zero(M.cols(), M.rows());
    }
    
    // Auto threshold if not specified
    double eps = threshold;
    if (threshold <= 0.0) {
        eps = std::max(M.rows(), M.cols()) * sv[0] * std::numeric_limits<double>::epsilon();
        eps = std::max(eps, 1e-15);
    }
    
    // Compute reciprocal with thresholding
    Eigen::VectorXd inv_sv(sv.size());
    for (int i = 0; i < sv.size(); ++i) {
        inv_sv[i] = (sv[i] > eps) ? 1.0 / sv[i] : 0.0;
    }
    
    return svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
}

// Helper function to compute robust Cholesky decomposition with multiple fallback strategies
bool VisualIntegrity::computeRobustCholesky(const Eigen::MatrixXd& A, Eigen::LLT<Eigen::MatrixXd>& llt_out, double& used_damping) {
    int N_cols = A.rows();
    
    // Strategy 1: Try standard Cholesky with adaptive damping
    double max_diag = 0.0;
    int max_index = 0;
    for (int i = 0; i < N_cols; ++i) {
        if (std::abs(A(i, i)) > max_diag) max_index = i;
        max_diag = std::max(max_diag, std::abs(A(i, i)));
    }
    if (max_diag == 0.0) max_diag = 1.0;
    
    // Start with small damping and increase gradually until success
    double base_damping = 1e-12;
    double adaptive_damping = base_damping * N_cols; //base_damping * max_diag * N_cols
    LOG(INFO) << "   - max_diag: " << max_diag  << ", at " << max_index<< ", N_cols: " << N_cols;
    double start_damping = std::max(base_damping, adaptive_damping);
    
    // Try increasing damping factors: 1x, 10x, 100x, 1000x, 10000x
    std::vector<double> damping_factors = {1.0, 10.0, 100.0, 1000.0, 10000.0};
    
    for (double factor : damping_factors) {
        double damping = start_damping * factor * max_diag;
        Eigen::MatrixXd A_damped = A + damping * Eigen::MatrixXd::Identity(N_cols, N_cols);
        llt_out.compute(A_damped);
        
        if (llt_out.info() == Eigen::Success) {
            used_damping = damping;
            if (factor == 1.0) {
                LOG(INFO) << "   - Cholesky decomposition succeeded with adaptive damping: " << std::setprecision(6) << damping;
            } else {
                LOG(WARNING) << "   - Cholesky decomposition succeeded with increased damping (factor " << factor << "): " << damping;
            }
            return true;
        } else{
            LOG(ERROR) << "   - Cholesky decomposition failed with damping factor " << factor << ": " << damping;
        }
    }
    
    // Strategy 2: Eigenvalue Reconstruction
    LOG(WARNING) << "   - LLT failed with damping, applying Eigenvalue Reconstruction (SVD-like approach)";
    // 使用 SelfAdjointEigenSolver (针对对称矩阵优化，比 SVD 快)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    if (es.info() != Eigen::Success) {
        LOG(ERROR) << "   - Eigen decomposition failed!";
        return false;
    }

    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();

    double max_eig = eigenvalues.maxCoeff();
    double min_threshold = std::max(1e-8, max_eig * 1e-12); 

    bool modified = false;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_threshold) {
            eigenvalues(i) = min_threshold;
            modified = true;
        }
    }

    if (modified) {
        // A_new = V * D_clamped * V^T
        Eigen::MatrixXd A_reconstructed = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        
        llt_out.compute(A_reconstructed);
        
        if (llt_out.info() == Eigen::Success) {
            used_damping = 0.0;
            LOG(INFO) << "   - Successfully recovered using Eigenvalue Clamping, with eigenvalue threshold: "<< min_threshold;
            return true;
        }
    }
    
    LOG(ERROR) << "   - All decomposition attempts failed.";
    double condition_number = computeConditionNumber(A);
    LOG(ERROR) << "   - Matrix condition number: " << condition_number;
    LOG(ERROR) << "   - Matrix size: " << N_cols << "x" << N_cols;
    LOG(ERROR) << "   - Min diagonal: " << A.diagonal().minCoeff();
    LOG(ERROR) << "   - Max diagonal: " << A.diagonal().maxCoeff();
    
    used_damping = 0.0;
    return false;
}

std::vector<int> VisualIntegrity::filteroutSubsets(Eigen::MatrixXd& sigma,
                                                   Eigen::MatrixXd& bias,
                                                   Eigen::MatrixXd& sigma_ss,
                                                   Eigen::MatrixXd& bias_ss,
                                                   Eigen::MatrixXd& s1vec,
                                                   Eigen::MatrixXd& s2vec,
                                                   Eigen::MatrixXd& s3vec,
                                                   Eigen::MatrixXd& x,
                                                   Eigen::VectorXd& chi2,
                                                   std::vector<std::vector<int>>& subsets,
                                                   std::vector<double>& pap_subsets,
                                                   double& p_not_monitored)
{
    std::vector<int> idx;
    for(int i = 0; i < sigma.rows(); ++i)
    {
        if (!std::isnan(sigma(i,0)) && !std::isnan(sigma(i,1)) && !std::isnan(sigma(i,2)) && !std::isnan(sigma_ss(i,0)) && !std::isnan(sigma_ss(i,1)) && !std::isnan(sigma_ss(i,2)) )
        {
            idx.push_back(i);
        } else {
            LOG(WARNING) << "Warning: Excluding subset " << i 
                         << " due to invalid sigma values (sigma: [" << sigma(i,0) << ", " << sigma(i,1) << ", " << sigma(i,2) 
                         << "], sigma_ss: [" << sigma_ss(i,0) << ", " << sigma_ss(i,1) << ", " << sigma_ss(i,2) << "])";
        }
    }

    Eigen::MatrixXd sigma_new(idx.size(), sigma.cols());
    Eigen::MatrixXd bias_new(idx.size(), bias.cols());
    Eigen::MatrixXd sigma_ss_new(idx.size(), sigma_ss.cols());
    Eigen::MatrixXd bias_ss_new(idx.size(), bias_ss.cols());
    Eigen::MatrixXd s1vec_new(idx.size(), s1vec.cols());
    Eigen::MatrixXd s2vec_new(idx.size(), s2vec.cols());
    Eigen::MatrixXd s3vec_new(idx.size(), s3vec.cols());
    Eigen::MatrixXd x_new(idx.size(), x.cols());
    Eigen::VectorXd chi2_new(idx.size());
    std::vector<std::vector<int>> subsets_new(idx.size());
    std::vector<double> pap_subsets_new(idx.size());
    
    for (int i = 0; i < idx.size(); ++i) {
        sigma_new.row(i) = sigma.row(idx[i]);
        bias_new.row(i) = bias.row(idx[i]);
        sigma_ss_new.row(i) = sigma_ss.row(idx[i]);
        bias_ss_new.row(i) = bias_ss.row(idx[i]);
        s1vec_new.row(i) = s1vec.row(idx[i]);
        s2vec_new.row(i) = s2vec.row(idx[i]);
        s3vec_new.row(i) = s3vec.row(idx[i]);
        x_new.row(i) = x.row(idx[i]);
        chi2_new(i) = chi2(idx[i]);
        subsets_new[i] = subsets[idx[i]];
        pap_subsets_new[i] = pap_subsets[idx[i]];
    }
    
    sigma = sigma_new;
    bias = bias_new;
    sigma_ss = sigma_ss_new;
    bias_ss = bias_ss_new;
    s1vec = s1vec_new;
    s2vec = s2vec_new;
    s3vec = s3vec_new;
    x = x_new;
    chi2 = chi2_new;
    subsets = subsets_new;
    p_not_monitored = p_not_monitored + std::accumulate(pap_subsets.begin(), pap_subsets.end(), 0.0)
                    - std::accumulate(pap_subsets_new.begin(), pap_subsets_new.end(), 0.0);
    pap_subsets = pap_subsets_new;

    // // Smoothly compress sigma spikes with axis-specific limits.
    // scaleSigmaSpikesInColumn(sigma, 0, 90.0, 0.02);
    // scaleSigmaSpikesInColumn(sigma, 1, 95.0, 0.10);
    // scaleSigmaSpikesInColumn(sigma, 2, 95.0, 0.10);
    // scaleSigmaSpikesInColumn(sigma_ss, 0, 90.0, 0.02);
    // scaleSigmaSpikesInColumn(sigma_ss, 1, 95.0, 0.10);
    // scaleSigmaSpikesInColumn(sigma_ss, 2, 95.0, 0.10);

    return idx;
}

Eigen::MatrixXd VisualIntegrity::computeTestThresholds(const Eigen::MatrixXd& sigma_ss,
                                                       const Eigen::MatrixXd& bias_ss)
{
    int N_sets = sigma_ss.rows();
    if (N_sets <= 1) {
        return Eigen::MatrixXd::Zero(N_sets, 3);
    }

    boost::math::normal_distribution<double> normal_d(0.0, 1.0);
    
    // Allocation of PFA
    double Kfa_la = -boost::math::quantile(normal_d, 0.5 * options_.PFA_La / (N_sets - 1));
    double Kfa_lo = -boost::math::quantile(normal_d, 0.5 * options_.PFA_Lo / (N_sets - 1));
    double Kfa_vert = -boost::math::quantile(normal_d, 0.5 * options_.PFA_V / (N_sets - 1));
    
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(N_sets, 3);
    
    T.col(0).array() = Kfa_la * sigma_ss.col(0).array() + bias_ss.col(0).array();
    T.col(1).array() = Kfa_lo * sigma_ss.col(1).array() + bias_ss.col(1).array();
    T.col(2).array() = Kfa_vert * sigma_ss.col(2).array() + bias_ss.col(2).array();

    return T;
}

void VisualIntegrity::computePL(const Eigen::MatrixXd& sigma,
                                const Eigen::MatrixXd& bias,
                                const Eigen::MatrixXd& T,
                                const std::vector<double>& pap_subset,
                                double p_not_monitored,
                                double& VPL,
                                double& LaPL,
                                double& LoPL,
                                double& HPL)
{
    if (pap_subset.empty()) {
        VPL = std::numeric_limits<double>::quiet_NaN();
        LaPL = std::numeric_limits<double>::quiet_NaN();
        LoPL = std::numeric_limits<double>::quiet_NaN();
        HPL = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    Eigen::Map<const Eigen::VectorXd> p_fault_const(pap_subset.data(), pap_subset.size());
    Eigen::VectorXd p_fault = p_fault_const;
    p_fault(0) = 2; //Server for IR and PL computation, because 2Q(***) +　Q(***)  

    // Allocation of PHMI
    double phmi_vert = options_.PHMI_V * (1.0 - p_not_monitored / options_.PHMI);
    double phmi_la = options_.PHMI_La * (1.0 - p_not_monitored / options_.PHMI);
    double phmi_lo = options_.PHMI_Lo * (1.0 - p_not_monitored / options_.PHMI);
    
    VPL = computeVPL(sigma.col(2), bias.col(2), T.col(2), p_fault, phmi_vert);
    LaPL = computeVPL(sigma.col(0), bias.col(0), T.col(0), p_fault, phmi_la);
    LoPL = computeVPL(sigma.col(1), bias.col(1), T.col(1), p_fault, phmi_lo);
    HPL = std::sqrt(LaPL*LaPL + LoPL*LoPL);
}

double VisualIntegrity::computeVPL(const Eigen::VectorXd& sigma_in,
                                   const Eigen::VectorXd& bias_in,
                                   const Eigen::VectorXd& T_in,
                                   const Eigen::VectorXd& p_fault_in,
                                   double phmi)
{
    // Make copies to modify
    Eigen::VectorXd sigma = sigma_in;
    Eigen::VectorXd bias = bias_in;
    Eigen::VectorXd T = T_in;
    Eigen::VectorXd p_fault = p_fault_in;
    double PL_TOL = options_.PL_TOL;

    const double MAX_ITERATION = 10;
    Eigen::VectorXd alloc_max = Eigen::VectorXd::Ones(sigma.rows()); 

    //Exclude sigmas that are inf and evaluate their integrity contribution.
    std::vector<int> index_Inf;
    std::vector<int> index_Fin;
    double p_not_monitorable = 0;
    for (int i = 0; i < sigma.rows(); ++i)
    {   
        if (sigma(i) == std::numeric_limits<double>::quiet_NaN())
        {
            index_Inf.push_back(i);
            p_not_monitorable += p_fault(i);
        }
        else{
            index_Fin.push_back(i);
        }
    } 

    if (p_not_monitorable >= phmi)
    {
        double VPL = std::numeric_limits<double>::quiet_NaN();
        return VPL;
    }

    Eigen::VectorXd sigma_new = Eigen::VectorXd::Ones(index_Fin.size()) * std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd bias_new = Eigen::VectorXd::Ones(index_Fin.size()) * std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd T_new = Eigen::VectorXd::Ones(index_Fin.size()) * std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd p_fault_new = Eigen::VectorXd::Ones(index_Fin.size()) * std::numeric_limits<double>::quiet_NaN();
    
    for (int i = 0, k = 0; i < index_Fin.size(); ++i)
    {
        sigma_new(k) = sigma(index_Fin[i]);
        bias_new(k) = bias(index_Fin[i]);
        T_new(k) = T(index_Fin[i]);
        p_fault_new(k) = p_fault(index_Fin[i]); 
        ++k;       
    }
    sigma = sigma_new; bias = bias_new; T = T_new; p_fault = p_fault_new;
    phmi = phmi - p_not_monitorable;

    //determine the lower bound on VPL 
    Eigen::VectorXd phmi_right_low = p_fault;
    Eigen::VectorXd Klow = p_fault;
    boost::math::normal_distribution<double> normal_d(0.0, 1.0); 
    for (int i = 0; i < Klow.rows() - 1; ++i)
    {
        phmi_right_low(i) =((phmi / (p_fault(i)  * alloc_max(i))) > 1 )?  1 : (phmi / (p_fault(i) * alloc_max(i)));
        if (phmi_right_low(i) < 0) phmi_right_low(i) = 0; // Safety check

        if(phmi_right_low(i) >= 1.0 - 1e-9)
        {
            Klow(i) = -std::numeric_limits<double>::quiet_NaN();
        }
        else
        {
            Klow(i) = - boost::math::quantile(normal_d, phmi_right_low(i));
        }
    }
    Klow.array() = T.array() + bias.array() + Klow.array() * sigma.array(); 
    double VPL_low = Klow.maxCoeff();

    //determine the upper bound on VPL 
    Eigen::VectorXd phmi_right_high = p_fault;
    phmi_right_high.array() = phmi / (sigma.rows() * p_fault.array());
    Eigen::VectorXd Khigh = p_fault;
    for (int i = 0; i < Khigh.rows();++i)
    {
        if (phmi_right_high(i) < 0) phmi_right_high(i) = 0; // Safety check

        // Handle the case where probability is 1.0 (quantile is infinity)
        if (phmi_right_high(i) >= 1.0 - 1e-9) {
             Khigh(i) = 0.0; // Or a very small number, effectively no protection needed from noise if risk budget is huge
        } else {
             Khigh(i) = - boost::math::quantile(normal_d, phmi_right_high(i));
        }
        
        if(Khigh(i) < 0) Khigh(i) = 0; 
    }
    Khigh.array() = T.array() + bias.array() + Khigh.array() * sigma.array(); 
    double VPL_high = Khigh.maxCoeff();

    //compute logarithm of phmi
    double log10phmi = std::log10(phmi);
    
    int count = 0;
    Eigen::VectorXd TbVs = Eigen::VectorXd::Zero(sigma.rows());
    while (((VPL_high - VPL_low) > PL_TOL) && (count < MAX_ITERATION))
    {
        ++count;
        double VPL_half = (VPL_high + VPL_low) / 2;

        double sum = 0;
        for (int i = 0; i < TbVs.rows(); ++i)
        {
            TbVs(i) = boost::math::cdf(normal_d, (T(i) + bias(i) - VPL_half) / sigma(i));
            if(TbVs(i) > 0.5) TbVs(i) = 1;
            if(TbVs(i) > alloc_max(i)) TbVs(i) = alloc_max(i);
            sum += p_fault(i) * TbVs(i);
        }
        double cdfhalf = std::log10(sum);
        if (cdfhalf > log10phmi) VPL_low = VPL_half;
        else VPL_high = VPL_half;
    }
    double VPL = VPL_high;
    return VPL;
}

double VisualIntegrity::computeIR(const Eigen::MatrixXd& sigma,
                                  const Eigen::MatrixXd& bias,
                                  const Eigen::MatrixXd& T,
                                  const std::vector<double>& pap_subset,
                                  double p_not_monitored)
{
    if (pap_subset.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::Map<const Eigen::VectorXd> p_fault_const(pap_subset.data(), pap_subset.size());
    Eigen::VectorXd p_fault = p_fault_const;
    p_fault(0) = 2; //Server for IR and PL computation, because 2Q(***) +　Q(***)  
    
    // T.row(0) = Eigen::MatrixXd::Zero(1,T.cols()); // T is const, cannot modify. But T(0) should be 0 anyway from computeTestThresholds if sigma_ss(0) is 0.
    // Actually computeTestThresholds sets T(0) to 0? No, it skips i=0. So T(0) is 0.
    
    Eigen::Vector3d AL(options_.HAL, options_.HAL/std::sqrt(2), options_.HAL/std::sqrt(2)); 
    Eigen::Vector3d IR_vec = Eigen::Vector3d::Zero();
    boost::math::normal_distribution<double> normal_d(0,1.0);
    
    for(int q = 0; q < 3; ++q)
    {
        for(int i = 0; i < sigma.rows(); ++i)
        {
            if(!std::isfinite(T(i,q)) || !std::isfinite(bias(i,q)) || !std::isfinite(sigma(i,q))) continue;
            IR_vec(q) += p_fault(i) * (1 - boost::math::cdf(normal_d, ((AL(q) - T(i,q) - bias(i,q)) / sigma(i,q))));
        }
    }

    return (IR_vec(0) + IR_vec(1) + IR_vec(2));
}

/*
Determine maximum simultanous faults need to monitor.
p: the probability of event including system
P_THRES: the probability to protect high probability to monitor
*/
int VisualIntegrity::determineNfaultmax(const std::vector<double>& p, double P_THRES)
{
    size_t n_p = p.size();
    size_t r = 0;
    double p_not_monitored = 1.0; 
    double pnofault_ = 1.0;
    std::vector<double> p_divisor;
    for(size_t i = 0; i < n_p;++i){
            p_divisor.push_back(p[i]/(1.0-p[i]));
            pnofault_ *= (1.0-p[i]);
    }

    while ((p_not_monitored > P_THRES)&&(r <= n_p))
    {
        r = r + 1;
 
        if(r <= 0){
            p_not_monitored = 1;
        }
        if(r == 1){
            p_not_monitored = 1 - pnofault_;
        }
        if(r == 2){
            double pmore = 0.0;
            for(size_t i = 0; i < n_p;++i){
                 pmore += p_divisor[i];
            }
            p_not_monitored = 1 - pnofault_ - pnofault_ * pmore;
        }
        if( r == 3){
            double pmore,pmore12= 0.0;
            for(size_t i = 0; i < n_p; ++i){
                pmore += p_divisor[i]; 
                for(size_t j = 0; j < i; ++j){
                    pmore12 += p_divisor[j] * p_divisor[i];
                }
            }
            p_not_monitored = 1 - pnofault_ - pnofault_ * pmore - pnofault_ * pmore12;
        }
        if(r >= 4){
            double sum_p = 0;
            for(size_t i = 0; i < n_p;++i){
                 sum_p += p[i];
            }
            double rr = 1;
            for(int i = 1;i<(r+1);++i){
                rr *= i;
            }
            p_not_monitored = std::pow(sum_p,r)/rr;
        }
    }

    int N_fault_max_ = r - 1;
    return N_fault_max_;
}

std::vector<std::vector<int>> VisualIntegrity::determine_k_subsets(int n, int k) 
{
  std::vector<std::vector<int>> subsets;
  
  if (k == 0) {
    std::vector<int> empty(n, 0);
    subsets.push_back(empty);
    return subsets;
  }
  
  if (k == 1) {
    for (int i = 0; i < n; i++) {
      std::vector<int> single(n, 0);
      single[i] = 1;
      subsets.push_back(single);
    }
    return subsets;
  }

  for (int i = 0; i <= n - k; i++) {
    std::vector<std::vector<int>> prev = determine_k_subsets(n - i - 1, k - 1);
    for (auto subset : prev) {
      std::vector<int> newSubset(n, 0);
      for (int j = 0; j < i; j++) newSubset[j] = 0;
      newSubset[i] = 1;
      for (int j = i + 1; j < n; j++) newSubset[j] = subset[j - i - 1];
      subsets.push_back(newSubset);
    }
  }
  return subsets;
}

int VisualIntegrity::nchoosek(int n, int k)
{   
    if(k > n - k){
        k = n - k;
    }
    int result = 1;
    for(int i = 0; i < k; ++i){
        result *=(n - i);
        result /=(i + 1);
    }
    return result;
}


// Helper function to read options from a stream
void VisualIntegrity::deserializeOptions(VisualIntegrityOptions& options, std::ifstream& in) {

    // integrity options
    in.read(reinterpret_cast<char*>(&options.enable), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.post_processing), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.snapshot_freq), sizeof(double));
    size_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));                  
    if (len > 4096) {
        LOG(ERROR) << "Invalid snapshot_file string length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.snapshot_file.resize(len);                                        
    if (len > 0) in.read(&options.snapshot_file[0], len);    

    // integrity_support_message
    in.read(reinterpret_cast<char*>(&options.use_complex_visual_cov), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.use_complex_gnss_cov), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.use_complex_imu_cov), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.use_complex_others_cov), sizeof(bool));    
    in.read(reinterpret_cast<char*>(&options.simple_visual_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.simple_gnss_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.simple_imu_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.simple_others_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.sigma_pixel), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.visual_prior_fault_probability), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.visual_meas_dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 1024) {
        LOG(ERROR) << "Invalid overbounding_func length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.overbounding_func.resize(len);
    if (len > 0) in.read(&options.overbounding_func[0], len);
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 10000) {
        LOG(ERROR) << "Invalid overbounding_parameters length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.overbounding_parameters.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.overbounding_parameters.data()), len * sizeof(double));
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 1024) {
        LOG(ERROR) << "Invalid normal_func length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.normal_func.resize(len);
    if (len > 0) in.read(&options.normal_func[0], len);
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 10000) {
        LOG(ERROR) << "Invalid normal_parameters length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.normal_parameters.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.normal_parameters.data()), len * sizeof(double));


    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid gnss_sigma_ura length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.gnss_sigma_ura.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.gnss_sigma_ura.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid gnss_sigma_ure length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.gnss_sigma_ure.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.gnss_sigma_ure.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid gnss_b_nom length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.gnss_b_nom.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.gnss_b_nom.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid gnss_sat_prior_fault_probability length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.gnss_sat_prior_fault_probability.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.gnss_sat_prior_fault_probability.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid gnss_const_prior_fault_probability length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.gnss_const_prior_fault_probability.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.gnss_const_prior_fault_probability.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&options.imu_prior_fault_probability), sizeof(double));

    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 32) {
        LOG(ERROR) << "Invalid user_F length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.user_F.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.user_F.data()), len * sizeof(double));

    in.read(reinterpret_cast<char*>(&options.user_Rr), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.user_a_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.user_b_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.doppler_c_sigma), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.rho_max), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.psi_user_deg), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.k_mp), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.tau_mp), sizeof(double));


    // navigation_requirements
    in.read(reinterpret_cast<char*>(&options.PHMI), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PHMI_La), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PHMI_Lo), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PHMI_V), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PFA), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PFA_La), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PFA_Lo), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PFA_V), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.HAL), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.VAL), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.LaAL), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.LoAL), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.P_THRES), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.Fc_THRES), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.PL_TOL), sizeof(double));
}

// Helper function to write options to a stream
void VisualIntegrity::serializeOptions(const VisualIntegrityOptions& options, std::ofstream& out) {
    
    // integrity options
    out.write(reinterpret_cast<const char*>(&options.enable), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.post_processing), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.snapshot_freq), sizeof(double));
    size_t snap_file_len = options.snapshot_file.size();
    out.write(reinterpret_cast<const char*>(&snap_file_len), sizeof(size_t));
    out.write(options.snapshot_file.c_str(), snap_file_len); 

    // integrity_support_message
    out.write(reinterpret_cast<const char*>(&options.use_complex_visual_cov), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.use_complex_gnss_cov), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.use_complex_imu_cov), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.use_complex_others_cov), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.simple_visual_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.simple_gnss_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.simple_imu_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.simple_others_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.sigma_pixel), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.visual_prior_fault_probability), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.visual_meas_dim), sizeof(int));
    size_t ob_func_len = options.overbounding_func.size();
    out.write(reinterpret_cast<const char*>(&ob_func_len), sizeof(size_t));
    out.write(options.overbounding_func.c_str(), ob_func_len);
    size_t ob_params_len = options.overbounding_parameters.size();
    out.write(reinterpret_cast<const char*>(&ob_params_len), sizeof(size_t));
    if (ob_params_len > 0) {
        out.write(reinterpret_cast<const char*>(options.overbounding_parameters.data()), ob_params_len * sizeof(double));
    }
    size_t norm_func_len = options.normal_func.size();
    out.write(reinterpret_cast<const char*>(&norm_func_len), sizeof(size_t));
    out.write(options.normal_func.c_str(), norm_func_len);
    size_t norm_params_len = options.normal_parameters.size();
    out.write(reinterpret_cast<const char*>(&norm_params_len), sizeof(size_t));
    if (norm_params_len > 0) {
        out.write(reinterpret_cast<const char*>(options.normal_parameters.data()), norm_params_len * sizeof(double));
    }


    size_t ura_size = options.gnss_sigma_ura.size();
    out.write(reinterpret_cast<const char*>(&ura_size), sizeof(size_t));
    if (ura_size > 0) {
        out.write(reinterpret_cast<const char*>(options.gnss_sigma_ura.data()), ura_size * sizeof(double));
    }

    size_t ure_size = options.gnss_sigma_ure.size();
    out.write(reinterpret_cast<const char*>(&ure_size), sizeof(size_t));
    if (ure_size > 0) {
        out.write(reinterpret_cast<const char*>(options.gnss_sigma_ure.data()), ure_size * sizeof(double));
    }

    size_t gnss_b_nom_size = options.gnss_b_nom.size();
    out.write(reinterpret_cast<const char*>(&gnss_b_nom_size), sizeof(size_t));
    if (gnss_b_nom_size > 0) {
        out.write(reinterpret_cast<const char*>(options.gnss_b_nom.data()), gnss_b_nom_size * sizeof(double));
    }

    size_t gnss_sat_prior_size = options.gnss_sat_prior_fault_probability.size();
    out.write(reinterpret_cast<const char*>(&gnss_sat_prior_size), sizeof(size_t));
    if (gnss_sat_prior_size > 0) {
        out.write(reinterpret_cast<const char*>(options.gnss_sat_prior_fault_probability.data()), gnss_sat_prior_size * sizeof(double));
    }

    size_t gnss_const_prior_size = options.gnss_const_prior_fault_probability.size();
    out.write(reinterpret_cast<const char*>(&gnss_const_prior_size), sizeof(size_t));
    if (gnss_const_prior_size > 0) {
        out.write(reinterpret_cast<const char*>(options.gnss_const_prior_fault_probability.data()), gnss_const_prior_size * sizeof(double));
    }

    out.write(reinterpret_cast<const char*>(&options.imu_prior_fault_probability), sizeof(double));

    size_t user_f_size = options.user_F.size();
    out.write(reinterpret_cast<const char*>(&user_f_size), sizeof(size_t));
    if (user_f_size > 0) {
        out.write(reinterpret_cast<const char*>(options.user_F.data()), user_f_size * sizeof(double));
    }

    out.write(reinterpret_cast<const char*>(&options.user_Rr), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.user_a_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.user_b_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.doppler_c_sigma), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.rho_max), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.psi_user_deg), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.k_mp), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.tau_mp), sizeof(double));

    // navigation_requirements
    out.write(reinterpret_cast<const char*>(&options.PHMI), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PHMI_La), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PHMI_Lo), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PHMI_V), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PFA), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PFA_La), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PFA_Lo), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PFA_V), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.HAL), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.VAL), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.LaAL), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.LoAL), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.P_THRES), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.Fc_THRES), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.PL_TOL), sizeof(double));
}

void VisualIntegrity::serializeSnapshot(const IntegritySnapshot& snapshot, std::ofstream& out) {
    // timestamp
    out.write(reinterpret_cast<const char*>(&snapshot.timestamp), sizeof(double));
    
    // J_all
    long rows = snapshot.J_all.rows();
    long cols = snapshot.J_all.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(long));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(long));
    out.write(reinterpret_cast<const char*>(snapshot.J_all.data()), rows * cols * sizeof(double));

    // r_all
    long size = snapshot.r_all.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(long));
    out.write(reinterpret_cast<const char*>(snapshot.r_all.data()), size * sizeof(double));

    // sig2_int
    rows = snapshot.sig2_int.rows();
    cols = snapshot.sig2_int.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(long));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(long));
    out.write(reinterpret_cast<const char*>(snapshot.sig2_int.data()), rows * cols * sizeof(double));

    // sig2_acc
    rows = snapshot.sig2_acc.rows();
    cols = snapshot.sig2_acc.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(long));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(long));
    out.write(reinterpret_cast<const char*>(snapshot.sig2_acc.data()), rows * cols * sizeof(double));

    // curr_lm_to_J_rows
    size_t map_size = snapshot.curr_lm_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_lm_to_J_rows) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // lm_to_J_rows
    map_size = snapshot.lm_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.lm_to_J_rows) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // curr_lm_to_object_ids
    map_size = snapshot.curr_lm_to_object_ids.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_lm_to_object_ids) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        int val = pair.second;
        out.write(reinterpret_cast<const char*>(&val), sizeof(int));
    }

    // lm_to_object_ids
    map_size = snapshot.lm_to_object_ids.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.lm_to_object_ids) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        int val = pair.second;
        out.write(reinterpret_cast<const char*>(&val), sizeof(int));
    }

    // curr_sat_to_J_rows
    map_size = snapshot.curr_sat_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_sat_to_J_rows) {
        size_t key_len = pair.first.size();
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(size_t));
        out.write(pair.first.data(), key_len);
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // sat_to_J_rows
    map_size = snapshot.sat_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.sat_to_J_rows) {
        size_t key_len = pair.first.size();
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(size_t));
        out.write(pair.first.data(), key_len);
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // curr_imu_to_J_rows
    map_size = snapshot.curr_imu_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_imu_to_J_rows) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // imu_to_J_rows
    map_size = snapshot.imu_to_J_rows.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.imu_to_J_rows) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint64_t));
        size_t vec_size = pair.second.size();
        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
    }

    // curr_pose_J_cols
    size_t vec_size = snapshot.curr_pose_J_cols.size();
    out.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(snapshot.curr_pose_J_cols.data()), vec_size * sizeof(int));

}
double VisualIntegrity::computeConditionNumber(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(svd.singularValues().size()-1);
    return sigma_max / sigma_min;
}

void VisualIntegrity::deserializeSnapshot(IntegritySnapshot& snapshot, std::ifstream& in) {
    // timestamp
    in.read(reinterpret_cast<char*>(&snapshot.timestamp), sizeof(double));
    if (in.fail()) return;

    // J_all
    long rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(long));
    in.read(reinterpret_cast<char*>(&cols), sizeof(long));
    if (rows < 0 || rows > 100000 || cols < 0 || cols > 10000) {
        LOG(ERROR) << "Invalid J_all dimensions: " << rows << "x" << cols;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.J_all.resize(rows, cols);
    in.read(reinterpret_cast<char*>(snapshot.J_all.data()), rows * cols * sizeof(double));

    // r_all
    long size;
    in.read(reinterpret_cast<char*>(&size), sizeof(long));
    if (size < 0 || size > 100000) {
        LOG(ERROR) << "Invalid r_all size: " << size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.r_all.resize(size);
    in.read(reinterpret_cast<char*>(snapshot.r_all.data()), size * sizeof(double));

    // sig2_int
    in.read(reinterpret_cast<char*>(&rows), sizeof(long));
    in.read(reinterpret_cast<char*>(&cols), sizeof(long));
    if (rows < 0 || rows > 100000 || cols < 0 || cols > 100000) {
        LOG(ERROR) << "Invalid sig2_int dimensions: " << rows << "x" << cols;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.sig2_int.resize(rows, cols);
    in.read(reinterpret_cast<char*>(snapshot.sig2_int.data()), rows * cols * sizeof(double));

    // sig2_acc
    in.read(reinterpret_cast<char*>(&rows), sizeof(long));
    in.read(reinterpret_cast<char*>(&cols), sizeof(long));
    if (rows < 0 || rows > 100000 || cols < 0 || cols > 100000) {
        LOG(ERROR) << "Invalid sig2_acc dimensions: " << rows << "x" << cols;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.sig2_acc.resize(rows, cols);
    in.read(reinterpret_cast<char*>(snapshot.sig2_acc.data()), rows * cols * sizeof(double));

    // curr_lm_to_J_rows
    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for curr_lm_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.curr_lm_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in curr_lm_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.curr_lm_to_J_rows[key] = vec;
    }

    // lm_to_J_rows
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (in.fail()) {
        in.clear();
        return;
    }
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for lm_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.lm_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in lm_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.lm_to_J_rows[key] = std::move(vec);
    }

    // curr_lm_to_object_ids
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    snapshot.curr_lm_to_object_ids.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        int val;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&val), sizeof(int));
        snapshot.curr_lm_to_object_ids[key] = val;
    }

    // lm_to_object_ids
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (in.fail()) {
        in.clear();
        return;
    }
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for lm_to_object_ids: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.lm_to_object_ids.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        int val;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&val), sizeof(int));
        snapshot.lm_to_object_ids[key] = val;
    }

    // curr_sat_to_J_rows
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for curr_sat_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.curr_sat_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        size_t key_len;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(size_t));
        if (key_len == 0 || key_len > 64) {
            LOG(ERROR) << "Invalid key_len in curr_sat_to_J_rows: " << key_len;
            in.setstate(std::ios::failbit);
            return;
        }
        std::string key(key_len, '\0');
        in.read(&key[0], key_len);
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in curr_sat_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.curr_sat_to_J_rows[key] = std::move(vec);
    }

    // sat_to_J_rows
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (in.fail()) {
        in.clear();
        return;
    }
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for sat_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.sat_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        size_t key_len;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(size_t));
        if (key_len == 0 || key_len > 64) {
            LOG(ERROR) << "Invalid key_len in sat_to_J_rows: " << key_len;
            in.setstate(std::ios::failbit);
            return;
        }
        std::string key(key_len, '\0');
        in.read(&key[0], key_len);
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in sat_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.sat_to_J_rows[key] = std::move(vec);
    }

    // curr_imu_to_J_rows
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for curr_imu_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.curr_imu_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in curr_imu_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.curr_imu_to_J_rows[key] = std::move(vec);
    }

    // imu_to_J_rows
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (in.fail()) {
        in.clear();
        return;
    }
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for imu_to_J_rows: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.imu_to_J_rows.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in imu_to_J_rows: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.imu_to_J_rows[key] = std::move(vec);
    }

    // curr_pose_J_cols
    size_t vec_size;
    in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
    snapshot.curr_pose_J_cols.resize(vec_size);
    in.read(reinterpret_cast<char*>(snapshot.curr_pose_J_cols.data()), vec_size * sizeof(int));

}




} // namespace gici
