/**
* @Function: Visual Integrity Monitoring using MHSS
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#include "gici/integrity/visual_integrity.h"
#include <atomic>

namespace gici {


VisualIntegrity::VisualIntegrity(const VisualIntegrityOptions& options)
    : options_(options), LaPL_(std::numeric_limits<double>::quiet_NaN()), LoPL_(std::numeric_limits<double>::quiet_NaN()), VPL_(std::numeric_limits<double>::quiet_NaN()), IR_(std::numeric_limits<double>::quiet_NaN())
{   
    is_first_ = true;
}


VisualIntegrity::~VisualIntegrity()
{
}


// The monitor function for real-time integrity monitoring
bool VisualIntegrity::monitor(const FramePtr& frame, const std::deque<State>& states, const Graph* graph, const PointMap& landmarks_map, size_t state_index)
{
    State state = states[state_index];
    timestamp_ = state.timestamp;
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return false;

    Eigen::MatrixXd J_all;
    Eigen::VectorXd r_all;
    Eigen::MatrixXd sig2_int;
    Eigen::MatrixXd sig2_acc;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols;
    std::map<uint64_t, std::vector<int>> curr_pose_to_J_cols;
    std::map<uint64_t, int> curr_lm_to_object_ids;
    std::vector<int> curr_pose_J_cols;

    if (!prepareLinearSystem(frame, states, state_index, graph, landmarks_map, 
                             J_all, r_all, sig2_int, sig2_acc, curr_lm_to_J_rows, curr_lm_to_J_cols, curr_lm_to_object_ids, curr_pose_to_J_cols, curr_pose_J_cols)) {
        return false;
    }
    

    computeIntegrityMetrics(J_all, r_all, sig2_int, sig2_acc, curr_lm_to_J_rows, curr_lm_to_J_cols, curr_lm_to_object_ids, curr_pose_J_cols);

    // Log results
    LOG(INFO) << std::scientific << std::setprecision(4)
              << "timestamp: " << timestamp_
              << ", LaPL: " << LaPL_ << " m"
              << ", LoPL: " << LoPL_ << " m"
              << ", VPL: " << VPL_ << " m";

    return (LaPL_ < options_.LaAL && LoPL_ < options_.LoAL && VPL_ < options_.VAL);
}

// Function to save integrity input information for post-processing
void VisualIntegrity::saveSnapshot(const FramePtr& frame, const std::deque<State>& states, const Graph* graph, const PointMap& landmarks_map, size_t state_index)
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

    State state = states[state_index];
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return;

    timestamp_ = state.timestamp;
    if (last_timestamp_ > 0 && (timestamp_ - last_timestamp_) < 1/options_.snapshot_freq) {
        LOG(INFO) << "The save snapshot frequency: " << options_.snapshot_freq << ", skipped timestamp: " << std::setprecision(6) << std::fixed << timestamp_;
        return;
    }
    last_timestamp_ = timestamp_;


    IntegritySnapshot snapshot;
    snapshot.timestamp = state.timestamp;

    if (!prepareLinearSystem(frame, states, state_index, graph, landmarks_map, 
                             snapshot.J_all, snapshot.r_all, snapshot.sig2_int, snapshot.sig2_acc, 
                             snapshot.curr_lm_to_J_rows, snapshot.curr_lm_to_J_cols, snapshot.curr_lm_to_object_ids,
                             snapshot.curr_pose_to_J_cols, snapshot.curr_pose_J_cols)) {
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
    if (is_first_ && ifs.peek() != EOF && !options_.yaml_options) {
        deserializeOptions(options_, ifs);
        LOG(INFO) << "Read options from snapshot file: " << filename;
        is_first_ = false;
    } else{
        VisualIntegrityOptions temp_opts;
        deserializeOptions(temp_opts, ifs);
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

        VPL_ = std::numeric_limits<double>::quiet_NaN();
        LaPL_ = std::numeric_limits<double>::quiet_NaN();
        LoPL_ = std::numeric_limits<double>::quiet_NaN();
        HPL_ = std::numeric_limits<double>::quiet_NaN();
        IR_ = 0;

        LOG(INFO) << std::fixed << std::setprecision(6) << "Timestamp: " << timestamp_;
        computeIntegrityMetrics(snapshot.J_all, snapshot.r_all, snapshot.sig2_int, snapshot.sig2_acc, snapshot.curr_lm_to_J_rows, snapshot.curr_lm_to_J_cols, snapshot.curr_lm_to_object_ids, snapshot.curr_pose_J_cols);

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
            double p_group = 1.0 - std::pow(1.0 - options_.prior_fault_probability, n_ms);
            p_prior_groups.push_back(p_group);
        }
        // Add probabilities for independent faults
        for (int i = 0; i < independent_faults; ++i) {
            p_prior_groups.push_back(options_.prior_fault_probability);
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

        std::vector<double> p_prior(num_meas, options_.prior_fault_probability);
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
            LOG(WARNING) << "===========================================================================";
            // LOG(WARNING) << "p_not_monitored_: " << p_not_monitored_;
            // LOG(WARNING) << "sigma_.size(): " << sigma_.rows() << " x " << sigma_.cols();
            // LOG(WARNING) << "Index\tSigma_1\tSigma_2\tSigma_3\tBias\tT_1\tT_2\tT_3\tP_fault";
            // for (int i = 0; i < sigma_.rows(); ++i) {
            //     LOG(WARNING) << i << "\t" 
            //             << sigma_(i, 0) << "\t" << sigma_(i, 1) << "\t" << sigma_(i, 2) << "\t"
            //             << bias_(i) << "\t" 
            //             << T_(i, 0) << "\t" << T_(i, 1) << "\t" << T_(i, 2) << "\t"
            //             << pap_subset_[i];
            // }
            // LOG(WARNING) << "===========================================================================";
        }

        // Free memory for the processed snapshot to avoid OOM
        snapshot.J_all.resize(0, 0);
        snapshot.r_all.resize(0);
        snapshot.sig2_int.resize(0, 0);
        snapshot.sig2_acc.resize(0, 0);
        snapshot.curr_lm_to_J_rows.clear();
        snapshot.curr_lm_to_J_cols.clear();
        snapshot.curr_pose_to_J_cols.clear();
        snapshot.curr_pose_J_cols.clear();

        // Generate timestamp for NMEA matching
        gtime_t t = gici::gnss_common::doubleToGtime(timestamp_);
        t = utc2gpst(t);
        t = gpst2utc(t);
        double ep[6];
        time2epoch(t, ep);
        
        double sod = ep[3] * 3600.0 + ep[4] * 60.0 + ep[5];
        results_list.push_back({sod, snapshot.timestamp, HPL_, VPL_, LaPL_, LoPL_, IR_});
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

    for (auto& l : lines) {
        // Check if line is $..IM and contains timestamp
        if (l.size() > 6 && l.substr(3, 3) == "IM,") {
            // Extract timestamp
            size_t first_comma = l.find(',');
            size_t second_comma = l.find(',', first_comma + 1);
            if (first_comma != std::string::npos && second_comma != std::string::npos) {
                std::string ts_str = l.substr(first_comma + 1, second_comma - first_comma - 1);
                
                try {
                    if (ts_str.size() >= 6) {
                        double h = std::stod(ts_str.substr(0, 2));
                        double m = std::stod(ts_str.substr(2, 2));
                        double s = std::stod(ts_str.substr(4));
                        double nmea_sod = h * 3600.0 + m * 60.0 + s;

                        for (const auto& item : results_list) {
                            if (std::abs(item.sod - nmea_sod) < 0.1) {
                                // Found match! Replace line
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
                                LOG(INFO) << "Updated NMEA line for timestamp " << ts_str;
                                break; // Stop searching for this line
                            }
                        }
                    }
                } catch (...) {
                    // Ignore parsing errors
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

        const double MAX_MERGE_DIST_SQ = 200.0 * 200.0;
        const int SMALL_CLUSTER_THRESHOLD = 2;
        const int LARGE_CLUSTER_THRESHOLD = 4;
        const int MIN_TOTAL_IDS_THRESHOLD = 18;

        // 1. Handling Large Clusters
        // Use a multi-pass approach to stabilize offloading
        bool unstable = true;
        int pass = 0;
        const int MAX_PASSES = 5;
        
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
                    
                    // Calculate Centroid
                    double sum_x = 0, sum_y = 0;
                    for(size_t idx : members) { sum_x += points[idx].x; sum_y += points[idx].y; }
                    double mean_x = sum_x / members.size();
                    double mean_y = sum_y / members.size();
                    
                    // Find seed (farthest from mean)
                    size_t seed_idx = members[0];
                    double max_dummy_dist = -1.0;
                    for(size_t idx : members) {
                         double d = (points[idx].x - mean_x)*(points[idx].x - mean_x) + (points[idx].y - mean_y)*(points[idx].y - mean_y);
                         if(d > max_dummy_dist) { max_dummy_dist = d; seed_idx = idx; }
                    }
                    
                    // Sort members by distance to seed
                    std::vector<std::pair<double, size_t>> dists;
                    for(size_t idx : members) {
                        double d = (points[idx].x - points[seed_idx].x)*(points[idx].x - points[seed_idx].x) + 
                                   (points[idx].y - points[seed_idx].y)*(points[idx].y - points[seed_idx].y);
                        dists.push_back({d, idx});
                    }
                    std::sort(dists.begin(), dists.end());
                    
                    // Assign half (closest to seed) to NEW ID
                    int new_id = ++max_id;
                    size_t split_count = members.size() / 2; 
                    // Ensure at least one point stays? yes members.size() > 8 implies split > 4
                    
                    for(size_t k = 0; k < split_count; ++k) {
                        points[dists[k].second].id = new_id;
                    }
                    
                    unstable = true; 
                    // Break outer loop to rebuild map? Or just continue? 
                    // Rebuilding is safer because we introduced a new ID
                    // But we can just continue to next cluster in the map
                }
            }
        }

        // 2. Handling Small Clusters
        // Only if we have enough total IDs
        std::map<int, int> counts;
        for (const auto& p : points) if (p.id >= 0) counts[p.id]++;
        
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

void VisualIntegrity::saveDebugImage(const FramePtr& frame, const PointMap& landmarks_map, const std::string& identifier) 
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
    std::string save_path = "/home/syl/GICI-IM/results/debug/debug_" + identifier + ".png";
    cv::imwrite(save_path, img_color);
    LOG(INFO) << "Saved debug image to: " << save_path;
}

bool VisualIntegrity::prepareLinearSystem(const FramePtr& frame, 
                             const std::deque<State>& states, 
                             size_t state_index, 
                             const Graph* graph, 
                             const PointMap& landmarks_map,
                             Eigen::MatrixXd& J_all, 
                             Eigen::VectorXd& r_all, 
                             Eigen::MatrixXd&  sig2_int,
                             Eigen::MatrixXd&  sig2_acc,
                             std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                             std::map<uint64_t, std::vector<int>>& curr_lm_to_J_cols,
                             std::map<uint64_t, int>& curr_lm_to_object_ids,
                             std::map<uint64_t, std::vector<int>>& curr_pose_to_J_cols,
                             std::vector<int>& curr_pose_J_cols)
{
    State state = states[state_index];
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return false;

    std::vector<std::pair<uint64_t, std::string>> row_ids_all;
    std::vector<std::pair<uint64_t, std::string>> col_ids_all;
    std::vector<std::pair<uint64_t, double>> pose_timestamps;
    std::vector<std::pair<uint64_t, int>> rows_curr;
    std::vector<std::pair<uint64_t, int>> cols_curr;   

    if (!extractFullLinearSystem(frame, states, state_index, graph, landmarks_map,
                             J_all, r_all, sig2_int, sig2_acc, row_ids_all, col_ids_all, pose_timestamps, rows_curr, cols_curr)) {
        LOG(ERROR) << "Failed to extract linear system.";
        return false;
    }
    
    saveDebugImage(frame, landmarks_map, std::to_string(state.timestamp));
    // saveEigenMatrixToFile(sig2_int, "/home/syl/GICI-IM/results/debug/sig2_int_output" + std::to_string(state.timestamp)  + ".txt");
    // saveEigenMatrixToFile(sig2_acc, "/home/syl/GICI-IM/results/debug/sig2_acc_output" + std::to_string(state.timestamp)  + ".txt");
    // saveFactorGraphDot(graph, state.id.asInteger(), pose_timestamps, "/home/syl/GICI-IM/results/factor_graph.dot");
    // printJacobianInfo(J_all, r_all, row_ids_all, col_ids_all, rows_curr, cols_curr, pose_timestamps, "/home/syl/GICI-IM/results/jacobian_visualization.txt");

    extractLandmarkRelatedRowsCols(frame, landmarks_map, row_ids_all, cols_curr, curr_lm_to_J_rows, curr_lm_to_J_cols, curr_lm_to_object_ids);
    extractPoseRelatedRowsCols(state.id.asInteger(), cols_curr, curr_pose_to_J_cols, curr_pose_J_cols); 
    
    return true;
}

bool VisualIntegrity::computeIntegrityMetrics(const Eigen::MatrixXd& J_all,
                                 const Eigen::VectorXd& r_all,
                                 const Eigen::MatrixXd& sig2_int,
                                 const Eigen::MatrixXd& sig2_acc,
                                 const std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                                 const std::map<uint64_t, std::vector<int>>& curr_lm_to_J_cols,
                                 const std::map<uint64_t, int>& curr_lm_to_object_ids,
                                 const std::vector<int>& curr_pose_J_cols)
{
    subsets_.clear();
    pap_subset_.clear();
    p_not_monitored_ = 0;

    std::vector<uint64_t> curr_lm_ids;

    int N_meas = curr_lm_to_J_rows.size();
    if (N_meas < 6) {
        LOG(ERROR) << "Not enough measurements: " << N_meas;
        return false;
    }

    for (const auto& lm_rows : curr_lm_to_J_rows) {
        curr_lm_ids.push_back(lm_rows.first);
    }

    // 2. Define Prior Probabilities
    // Assuming independent faults for each feature with a fixed probability
    double p_feat_fault = options_.prior_fault_probability; 
    
    // Construct fault groups
    std::vector<std::vector<uint64_t>> fault_groups;
    std::vector<double> p_prior_groups;

    if (options_.use_segment) {
        std::map<int, std::vector<uint64_t>> object_groups;
        std::vector<uint64_t> independent_lms;

        for (uint64_t lm_id : curr_lm_ids) {
             int obj_id = -1;
             if (curr_lm_to_object_ids.count(lm_id)) {
                 obj_id = curr_lm_to_object_ids.at(lm_id);
             }
             if (obj_id >= 0) {
                 object_groups[obj_id].push_back(lm_id);
             } else {
                 independent_lms.push_back(lm_id);
             }
        }

        // Add object groups
        for (const auto& pair : object_groups) {
             fault_groups.push_back(pair.second);
             double p = 1.0 - std::pow(1.0 - p_feat_fault, pair.second.size());
             p_prior_groups.push_back(p);
        }
        // Add independent lms
        for (uint64_t lm_id : independent_lms) {
             fault_groups.push_back({lm_id});
             p_prior_groups.push_back(p_feat_fault);
        }
        LOG(INFO) << "Defined " << object_groups.size() << " object groups and " << independent_lms.size() << " independent measurements.";

    } else {
         for (uint64_t lm_id : curr_lm_ids) {
             fault_groups.push_back({lm_id});
             p_prior_groups.push_back(p_feat_fault);
         }
    }


    // 3. Determine Subsets
    determineSubsets(p_prior_groups, subsets_, pap_subset_, p_not_monitored_);
    CHECK_EQ(fault_groups.size(), subsets_[0].size());

    LOG(INFO) << "Total subsets to monitor: " << subsets_.size();

    // 4. Compute Subset Solutions
    computeSubsetSolution(J_all, r_all, sig2_int, sig2_acc, subsets_, fault_groups, curr_lm_to_J_rows, curr_lm_to_J_cols, curr_lm_ids, curr_pose_J_cols, sigma_, bias_, sigma_ss_, bias_ss_, s1vec_, s2vec_, s3vec_, x_, chi2_);

    // 5. Filter out unmonitorable subsets
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
    if (fault_detected) LOG(WARNING) << std::fixed << std::setprecision(6)<< "Fault detected num: " << fault_detected_num << ", for timestamp: " << timestamp_;
    if (!fault_detected)  LOG(INFO) << std::fixed << std::setprecision(6)<< "No fault detected for timestamp: " << timestamp_;

    // 8. Compute PL and IR
    computePL(sigma_, bias_, T_, pap_subset_, p_not_monitored_, VPL_, HPL_, LaPL_, LoPL_);
    IR_ = computeIR(sigma_, bias_, T_, pap_subset_, p_not_monitored_);

    return fault_detected;
}


bool VisualIntegrity::extractFullLinearSystem(const FramePtr& frame, const std::deque<State>& states, size_t state_index, const Graph* graph, const PointMap& landmarks_map,
                                              Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::MatrixXd& sig2_int, Eigen::MatrixXd& sig2_acc,
                                              std::vector<std::pair<uint64_t, std::string>>& row_ids_all, std::vector<std::pair<uint64_t, std::string>>& col_ids_all, std::vector<std::pair<uint64_t, double>>& pose_timestamps,
                                              std::vector<std::pair<uint64_t, int>>& rows_curr, std::vector<std::pair<uint64_t, int>>& cols_curr)
{
    
    
    uint64_t current_pose_id = states[state_index].id.asInteger();
    if (!graph->parameterBlockExists(current_pose_id)) return false;

    struct GenericResidualInfo {
        double timestamp;
        std::pair<uint64_t, std::string> row_id; 
        Eigen::VectorXd residual;
        double sig2_int;
        double sig2_acc;
        int cur_track;
        bool is_current_frame;
        uint64_t landmark_id;
        std::vector<std::pair<uint64_t, Eigen::MatrixXd>> jacobians; // ParamID, Jacobian
    };
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
                 BackendId pose_id = changeIdType(pb_id, IdType::cPose);
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

        std::string row_id_type = "Unknown";
        auto error_type = graph->errorInterfacePtr(residual_block_id)->typeInfo();
        row_id_type = kErrorToStr.at(error_type);
        GenericResidualInfo info;
        info.timestamp = timestamp;
        info.row_id = {reinterpret_cast<uint64_t>(residual_block_id), row_id_type};
        info.residual = Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals);
        info.is_current_frame = is_current;

        for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            if (jacobians[i] != nullptr) {
                int dim = parameter_blocks[i].second->minimalDimension();
                Eigen::MatrixXd J = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[i], num_residuals, dim);
                info.jacobians.push_back({parameter_blocks[i].first, J});
            }
        }

        info.sig2_int = options_.sigma_pixel * options_.sigma_pixel; // Default
        info.sig2_acc = options_.sigma_pixel * options_.sigma_pixel; // Default
        info.landmark_id = 0; // Default
        info.cur_track = -1;
        if (error_type == ErrorType::kReprojectionError && options_.overbounding_func != "none") {
            // Find landmark ID from parameter blocks
            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                BackendId pb_id(parameter_blocks[i].first);
                if (pb_id.type() == IdType::cLandmark) {
                    info.landmark_id = pb_id.asInteger();
                    break;
                }
            }
        }

        if (info.landmark_id != 0) {
            info.sig2_int = options_.sigma_pixel * options_.sigma_pixel;
            info.sig2_acc = options_.sigma_pixel * options_.sigma_pixel;
        } else{
            info.sig2_int = 1;
            info.sig2_acc = 1;
        }
        all_residuals.push_back(info);
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
    if (options_.overbounding_func == "dual_exp"){
        
        for(auto const& pair : lm_to_indices) {
            uint64_t lm_id = pair.first;
            if (landmarks_map.find(BackendId(lm_id)) == landmarks_map.end()) continue;
            const auto& landmark = landmarks_map.at(BackendId(lm_id));

            std::map<uint64_t, int> res_to_frame;
            for(const auto& obs : landmark.observations) res_to_frame[obs.second] = obs.first.frame_id;
            
            std::map<int, int> frame_to_track;
            for(size_t k = 0; k < landmark.point->obs_.size(); ++k) frame_to_track[landmark.point->obs_[k].frame_id] = k;

            for(size_t idx : pair.second) {
                uint64_t res_id = all_residuals[idx].row_id.first;
                if(res_to_frame.count(res_id)) {
                    int fid = res_to_frame[res_id];
                    if(frame_to_track.count(fid)) {
                        int cur_track = frame_to_track[fid];
                        all_residuals[idx].cur_track = cur_track;
                        all_residuals[idx].sig2_int += computeDualExpOverboundingSig2(options_.overbounding_parameters, 0, cur_track);
                        if (options_.normal_func != "none")  all_residuals[idx].sig2_acc += computeDualExpOverboundingSig2(options_.normal_parameters, 0, cur_track);
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

            for (const auto& pair : info.jacobians) { //cols_curr not include imu error to 上一帧
                int col = param_col_map[pair.first];
                J_all.block(current_row_idx, col, num_res, pair.second.cols()) = pair.second;
                if (info.is_current_frame) {
                    for (int r = 0; r < num_res; ++r) rows_curr.push_back(std::make_pair(info.row_id.first, current_row_idx + r));

                    bool add_curr = false;
                    BackendId parmid(pair.first);
                    if (parmid.type() == IdType::ImuStates) {
                        parmid = changeIdType(parmid, IdType::cPose);
                    }
                    if (parmid.asInteger() == current_pose_id || parmid.type() != IdType::cPose) {
                        add_curr = true;
                    }
                    if (add_curr) {
                        for (int c = 0; c < pair.second.cols(); ++c) cols_curr.push_back(std::make_pair(pair.first, col + c)); 
                    }
                }
            }
            current_row_idx += num_res;
        }
        
        if (options_.overbounding_func != "none") {
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
                        
                        sig2_int.block(row1, row2, n1, n2) = min_sig2_int * Eigen::MatrixXd::Identity(n1, n2);
                        sig2_int.block(row2, row1, n2, n1) = min_sig2_int * Eigen::MatrixXd::Identity(n2, n1);

                        if (options_.normal_func == "none") continue;
                        sig2_acc.block(row1, row2, n1, n2) = min_sig2_acc * Eigen::MatrixXd::Identity(n1, n2);
                        sig2_acc.block(row2, row1, n2, n1) = min_sig2_acc * Eigen::MatrixXd::Identity(n2, n1);
                    }
                }
            }
        }
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

void VisualIntegrity::extractLandmarkRelatedRowsCols(const FramePtr& frame, const PointMap& landmarks_map,
                                                  const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_rows,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_cols,
                                                  std::map<uint64_t, int>& landmark_object_ids)
{

    // Pre-build lookup maps for O(1) access
    std::unordered_map<uint64_t, std::vector<int>> resid_to_rows_map;
    for (size_t i = 0; i < row_ids_all.size(); ++i) {
        resid_to_rows_map[row_ids_all[i].first].push_back(static_cast<int>(i));
    }

    std::unordered_map<uint64_t, std::vector<int>> lm_to_cols_map;
    for (const auto& pair : cols_curr) {
        lm_to_cols_map[pair.first].push_back(pair.second);
    }

    std::vector<BalancePoint> current_frame_points;
    std::vector<uint64_t> current_frame_lm_ids;

    // Iterate through landmarks map once
    for (const auto& lm_pair : landmarks_map) {
        uint64_t lm_id = lm_pair.first.asInteger();

        // Check if this landmark is in our current columns of interest
        auto col_it = lm_to_cols_map.find(lm_id);
        if (col_it != lm_to_cols_map.end()) {
            // Store Column Indices
            landmark_observation_cols[lm_id] = col_it->second;
            CHECK_EQ(landmark_observation_cols[lm_id].size(), 3);

            // Store Row Indices for all observations of this landmark
            std::vector<int>& rows_vec = landmark_observation_rows[lm_id];
            for (const auto& obs_pair : lm_pair.second.observations) {
                uint64_t res_id = obs_pair.second;
                
                auto row_it = resid_to_rows_map.find(res_id);
                if (row_it != resid_to_rows_map.end()) {
                    rows_vec.insert(rows_vec.end(), row_it->second.begin(), row_it->second.end());
                }

                if (obs_pair.first.frame_id == frame->id()) {
                    size_t feat_idx_curr = obs_pair.first.keypoint_index_;
                    int obj_id_curr = -1;
                    if (feat_idx_curr < frame->object_id_vec_.size()) {
                        obj_id_curr = frame->object_id_vec_[feat_idx_curr];
                    }
                    current_frame_points.push_back({frame->px_vec_(0, feat_idx_curr), frame->px_vec_(1, feat_idx_curr), obj_id_curr});
                    current_frame_lm_ids.push_back(lm_id);
                    
                }else{
                    landmark_object_ids[lm_id] = -1;
                }
            }
        }
    }
    
    // Balance IDs
    balanceObjectIds(current_frame_points);

    // Apply balanced IDs
    for (size_t i = 0; i < current_frame_points.size(); ++i) {
        landmark_object_ids[current_frame_lm_ids[i]] = current_frame_points[i].id;
    }

    CHECK_EQ(landmark_observation_rows.size(), landmark_observation_cols.size());
}


void VisualIntegrity::extractPoseRelatedRowsCols(uint64_t current_pose_id,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& pose_related_cols,
                                                  std::vector<int>& curr_pose_J_cols)
{


    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if (pb_id.type() == IdType::cPose && pb_id.asInteger() == current_pose_id) {
            curr_pose_J_cols.push_back(cols_curr[i].second);
            pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
            if (curr_pose_J_cols.size() == 6) break; // Early exit if all expected columns are found
            
        }
    }

    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if (pb_id.type() == IdType::ImuStates) {
            pb_id = changeIdType(pb_id, IdType::cPose);
            if (pb_id.asInteger() == current_pose_id) {
                curr_pose_J_cols.push_back(cols_curr[i].second);
                pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
                if (curr_pose_J_cols.size() == 15) break; // Early exit if all expected columns are found
            }
        } 
    }

    CHECK_EQ(pose_related_cols.size(), 2);
    CHECK_EQ(curr_pose_J_cols.size(), 15);
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
    LOG(INFO) << "The maximum simultanous faults need to monitor = " << N_fault_max << ", in P_THRES = " << P_THRES;

    //Calculate the number of subsets.
    int N_used = N;
    int subsetsize = 0;
    for(int j = 0; j <= N_fault_max;++j){
        subsetsize = subsetsize + nchoosek((N_used),j);
    }

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
                                            const std::vector<std::vector<uint64_t>>& fault_groups,
                                            const std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows,
                                            const std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols,
                                            const std::vector<uint64_t>& curr_lm_ids,
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
    std::vector<std::vector<int>> lm_rows_cache(N_meas_curr);

    // Pre-cache rows for each fault group
    for(int j=0; j<N_meas_curr; ++j) {
        for (uint64_t lm_id : fault_groups[j]) {
            if (curr_lm_to_J_rows.count(lm_id)) {
                 const auto& rows = curr_lm_to_J_rows.at(lm_id);
                 lm_rows_cache[j].insert(lm_rows_cache[j].end(), rows.begin(), rows.end());
            }
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
        
        #pragma omp for schedule(dynamic, 16)  // Dynamic scheduling for load balancing
        for (int i = 1; i < N_sets; ++i) {
            rows_to_remove.clear();
            int lm_to_remove = 0;
            // Identify rows to remove based on current subset
            for (int j = 0; j < N_meas_curr; ++j) {
                if (subsets[i][j] == 0) { // 0 means fault/exclude
                    ++lm_to_remove;
                    const auto& rows = lm_rows_cache[j];
                    rows_to_remove.insert(rows_to_remove.end(), rows.begin(), rows.end());
                    // LOG(INFO) << "Subset " << i << ": Excluding landmark group " << j << " with " << rows.size() << " rows.";
                }
            }

            // Check observability roughly
            if (!rows_to_remove.empty() && (N_meas_curr - lm_to_remove < 6)) {
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
            Eigen::MatrixXd JP = J_rem * P_all; // n_rem x N_cols
            Eigen::MatrixXd W_rem_inv;
            if (W_is_diagonal) {
                W_rem_inv = W_rem.diagonal().cwiseInverse().asDiagonal();
            } else {
                W_rem_inv = robustInverse(W_rem);
            }
            
            Eigen::MatrixXd Middle = W_rem_inv - (JP * J_rem.transpose());
            Eigen::MatrixXd Kernel = robustInverse(Middle);
            Eigen::MatrixXd UpdateBlock = JP.transpose() * Kernel;
            Eigen::VectorXd b_sub = b_all - J_rem.transpose() * W_rem * r_rem;
            Eigen::VectorXd x_curr_full = P_all * b_sub + UpdateBlock * (JP * b_sub);

            // Compute S vectors and Sigma for all 3 dimensions
            for (int k = 0; k < 3; ++k) {
                int row_id = curr_pose_J_cols[k];
                x(i, k) = x_curr_full(row_id);
                // Compute specific row of P_sub: P_row + UpdateBlock_row * JP
                Eigen::RowVectorXd P_sub_row = P_all.row(row_id) + UpdateBlock.row(row_id) * JP;
                // Compute S vector: S = P_sub_row * JtW_all
                Eigen::VectorXd s_row = (P_sub_row * JtW_all).transpose();
                // Set S values for removed measurements to 0
                for(size_t r=0; r<rows_to_remove.size(); ++r) {
                    s_row(rows_to_remove[r]) = 0.0;
                }

                // Store S vectors
                if(k==0) s1vec.row(i) = s_row;
                if(k==1) s2vec.row(i) = s_row;
                if(k==2) s3vec.row(i) = s_row;
                Eigen::VectorXd ds = s_row - s_base[k];
                
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
                    LOG(WARNING) << "Debug Info: subset index = " << i << ", dimension = " << k << ", lm_to_remove = " << lm_to_remove;
                    LOG(WARNING) << "Debug Info: rows_to_remove size = " << rows_to_remove.size() << ", N_meas_curr = " << N_meas_curr;
                    LOG(WARNING) << "Debug Info: P_all norm = " << P_all.norm() << ", JtW_all norm = " << JtW_all.norm();
                    LOG(WARNING) << "Debug Info: JP norm = " << JP.norm() << ", W_rem_inv norm = " << W_rem_inv.norm();
                    LOG(WARNING) << "Debug Info: Middle norm = " << Middle.norm() << ", Kernel norm = " << Kernel.norm();
                    LOG(WARNING) << "Debug Info: UpdateBlock norm = " << UpdateBlock.norm() << ", b_sub norm = " << b_sub.norm();
                    LOG(WARNING) << "Debug Info: x_curr_full norm = " << x_curr_full.norm() << ", P_sub_row norm = " << P_sub_row.norm();
                    saveEigenMatrixToFile(sig2_int, "/home/syl/GICI-IM/results/debug/sig2_int_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
                    saveEigenMatrixToFile(sig2_acc, "/home/syl/GICI-IM/results/debug/sig2_acc_debug_" + std::to_string(timestamp_)  + "_" + std::to_string(i)+ ".txt");
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
    for (int i = 0; i < N_cols; ++i) {
        max_diag = std::max(max_diag, std::abs(A(i, i)));
    }
    if (max_diag == 0.0) max_diag = 1.0;
    
    // Start with small damping and increase gradually until success
    double base_damping = 1e-9;
    double adaptive_damping = base_damping * N_cols; //base_damping * max_diag * N_cols
    LOG(INFO) << "   - max_diag: " << max_diag << ", N_cols: " << N_cols;
    double start_damping = std::max(base_damping, adaptive_damping);
    
    // Try increasing damping factors: 1x, 10x, 100x, 1000x, 10000x
    std::vector<double> damping_factors = {1.0, 10.0, 100.0, 1000.0, 10000.0};
    
    for (double factor : damping_factors) {
        double damping = start_damping * factor;
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
    in.read(reinterpret_cast<char*>(&options.enable), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.sigma_pixel), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.prior_fault_probability), sizeof(double));
    in.read(reinterpret_cast<char*>(&options.use_segment), sizeof(bool));
    in.read(reinterpret_cast<char*>(&options.meas_dim), sizeof(int));
    
    // overbounding_func string
    size_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 1024) {
        LOG(ERROR) << "Invalid overbounding_func length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.overbounding_func.resize(len);
    if (len > 0) in.read(&options.overbounding_func[0], len);

    // overbounding_parameters vector
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 10000) {
        LOG(ERROR) << "Invalid overbounding_parameters length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.overbounding_parameters.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.overbounding_parameters.data()), len * sizeof(double));

    // normal_func string
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 1024) {
        LOG(ERROR) << "Invalid normal_func length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.normal_func.resize(len);
    if (len > 0) in.read(&options.normal_func[0], len);

    // normal_parameters vector
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 10000) {
        LOG(ERROR) << "Invalid normal_parameters length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.normal_parameters.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(options.normal_parameters.data()), len * sizeof(double));

    // Other doubles
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
    
    // snapshot_file string
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    if (len > 4096) {
        LOG(ERROR) << "Invalid snapshot_file string length: " << len;
        in.setstate(std::ios::failbit);
        return;
    }
    options.snapshot_file.resize(len);
    if (len > 0) in.read(&options.snapshot_file[0], len);
}

// Helper function to write options to a stream
void VisualIntegrity::serializeOptions(const VisualIntegrityOptions& options, std::ofstream& out) {
    out.write(reinterpret_cast<const char*>(&options.enable), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.sigma_pixel), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.prior_fault_probability), sizeof(double));
    out.write(reinterpret_cast<const char*>(&options.use_segment), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&options.meas_dim), sizeof(int));
    
    // overbounding_func string
    size_t ob_func_len = options.overbounding_func.size();
    out.write(reinterpret_cast<const char*>(&ob_func_len), sizeof(size_t));
    out.write(options.overbounding_func.c_str(), ob_func_len);

    // overbounding_parameters vector
    size_t ob_params_len = options.overbounding_parameters.size();
    out.write(reinterpret_cast<const char*>(&ob_params_len), sizeof(size_t));
    if (ob_params_len > 0) {
        out.write(reinterpret_cast<const char*>(options.overbounding_parameters.data()), ob_params_len * sizeof(double));
    }

    // normal_func string
    size_t norm_func_len = options.normal_func.size();
    out.write(reinterpret_cast<const char*>(&norm_func_len), sizeof(size_t));
    out.write(options.normal_func.c_str(), norm_func_len);

    // normal_parameters vector
    size_t norm_params_len = options.normal_parameters.size();
    out.write(reinterpret_cast<const char*>(&norm_params_len), sizeof(size_t));
    if (norm_params_len > 0) {
        out.write(reinterpret_cast<const char*>(options.normal_parameters.data()), norm_params_len * sizeof(double));
    }

    // Other doubles
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
    
    // snapshot_file string
    size_t snap_file_len = options.snapshot_file.size();
    out.write(reinterpret_cast<const char*>(&snap_file_len), sizeof(size_t));
    out.write(options.snapshot_file.c_str(), snap_file_len);
}

void VisualIntegrity::serializeSnapshot(const IntegritySnapshot& snapshot, std::ofstream& out) {
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

    // curr_lm_to_J_cols
    map_size = snapshot.curr_lm_to_J_cols.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_lm_to_J_cols) {
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

    // curr_pose_to_J_cols
    map_size = snapshot.curr_pose_to_J_cols.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& pair : snapshot.curr_pose_to_J_cols) {
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

    // curr_lm_to_J_cols
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    if (map_size > 1000000) {
        LOG(ERROR) << "Invalid map_size for curr_lm_to_J_cols: " << map_size;
        in.setstate(std::ios::failbit);
        return;
    }
    snapshot.curr_lm_to_J_cols.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        if (vec_size > 100000) {
            LOG(ERROR) << "Invalid vec_size in curr_lm_to_J_cols: " << vec_size;
            in.setstate(std::ios::failbit);
            return;
        }
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.curr_lm_to_J_cols[key] = vec;
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

    // curr_pose_to_J_cols
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    snapshot.curr_pose_to_J_cols.clear();
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t key;
        size_t vec_size;
        in.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
        std::vector<int> vec(vec_size);
        in.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));
        snapshot.curr_pose_to_J_cols[key] = vec;
    }

    // curr_pose_J_cols
    size_t vec_size;
    in.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
    snapshot.curr_pose_J_cols.resize(vec_size);
    in.read(reinterpret_cast<char*>(snapshot.curr_pose_J_cols.data()), vec_size * sizeof(int));

}


bool VisualIntegrity::extractLinearSystem(const FramePtr& frame, const State& state, const Graph* graph, const PointMap& landmarks_map,
                                          Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::MatrixXd& sig2_all,
                                          std::vector<uint64_t>& row_ids_all, std::vector<uint64_t>& col_ids_all,
                                          std::vector<std::pair<uint64_t, int>> rows_curr, std::vector<std::pair<uint64_t, int>> cols_curr)
{
    // Iterate over observations in the frame
    // Find corresponding residual blocks in the graph
    // Evaluate Jacobian w.r.t. the pose state

        
    // some examples in computeAndGetCovariance(states_[latest_state_index_]);
    // std::vector<size_t> parameter_block_ids;
    // BackendId id = state.id_in_graph;
    // parameter_block_ids.push_back(id.asInteger());
    // BackendId speed_and_bias_id = changeIdType(id, IdType::ImuStates);
    // parameter_block_ids.push_back(speed_and_bias_id.asInteger());
    // for (size_t i = 0; i < parameter_block_ids.size(); i++) {
    //     auto it = graph->id_to_parameter_block_map_.find(parameter_block_ids[i]);
    //     if (it == graph->id_to_parameter_block_map_.end()) {
    //         LOG(ERROR) << "Parameter block does not exist!";
    //         return false;
    //     }
    //     auto& parameter_block = it->second;
    //     LOG(INFO) << "Parameter Block ID: " << it->first
    //               << " Type: " << parameter_block->typeInfo()
    //               << " Dimension: " << parameter_block->dimension();
    // }

    // Eigen::MatrixXd covariance;
    // if (!const_cast<Graph*>(graph)->computeCovariance(parameter_block_ids, covariance)) {
    //     return false;
    // }
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return false;
    
    uint64_t current_pose_id = state.id.asInteger();
    if (!graph->parameterBlockExists(current_pose_id)) return false;

    struct ParamResidualInfo {
        uint64_t landmark_id;
        std::vector<uint64_t> pose_ids;
        std::vector<uint64_t> speed_bias_ids;
        Eigen::VectorXd residual;
        std::vector<Eigen::MatrixXd> J_poses;
        Eigen::MatrixXd J_landmark;
        std::vector<Eigen::MatrixXd> J_speed_biases;
        bool has_landmark_jacobian;
        double sig2;
        bool is_current_frame;
    };
    std::vector<ParamResidualInfo> all_residuals;
    
    // 1. Identify all involved poses (Current Pose + Poses observing landmarks in landmarks_map)
    std::set<uint64_t> involved_poses;
    involved_poses.insert(current_pose_id);

    ceres::Problem* problem = graph->problem().get();

    // Iterate landmarks_map to find involved poses
    for(auto it = landmarks_map.begin(); it != landmarks_map.end(); ++it)
    {
        const MapPoint& map_point = it->second;
        for (auto obs : map_point.observations) {
            ceres::ResidualBlockId residual_block_id = ceres::ResidualBlockId(obs.second);
            gici::Graph::ParameterBlockCollection parameter_blocks = graph->parameters(residual_block_id);
            
            for (const auto& pb : parameter_blocks) {
                BackendId pb_id(pb.first);
                if (pb_id.type() == IdType::cPose) {
                    involved_poses.insert(pb.first);
                    BackendId speed_and_bias_id = changeIdType(pb_id, IdType::ImuStates);
                    involved_poses.insert(speed_and_bias_id.asInteger());
                }
            }
        }
    }

    // 2. Collect all residuals for these poses
    std::set<ceres::ResidualBlockId> processed_residuals;
    
    for (uint64_t pose_id : involved_poses) {
        gici::Graph::ResidualBlockCollection residuals = graph->residuals(pose_id);
        
        // graph->printParameterBlockInfo(pose_id);
        
        for (const auto& res_spec : residuals) {
            ceres::ResidualBlockId residual_block_id = res_spec.residual_block_id;

            auto error_type = graph->errorInterfacePtr(residual_block_id)->typeInfo();
            if (error_type != ErrorType::kReprojectionError && error_type != ErrorType::kIMUError) {
                LOG(ERROR) << "Skipping non-visual residual block Type: " << kErrorToStr.at(error_type);
                continue;
            }
            
            if (processed_residuals.count(residual_block_id)) continue;
            processed_residuals.insert(residual_block_id);

            const ceres::CostFunction* cost_function = problem->GetCostFunctionForResidualBlock(residual_block_id);
            if (cost_function == nullptr) continue; 
            
            int num_residuals = cost_function->num_residuals();

            gici::Graph::ParameterBlockCollection parameter_blocks = graph->parameters(residual_block_id);
            
            // Identify Pose, Landmark, speed and bias parameter blocks
            std::vector<int> pose_indices;
            int landmark_idx = -1;
            std::vector<int> speed_bias_indices;
            
            std::vector<uint64_t> obs_pose_ids;
            uint64_t obs_landmark_id = 0;
            std::vector<uint64_t> obs_speed_bias_ids;
            
            std::vector<double> residuals_eval(num_residuals);
            std::vector<double*> jacobians(parameter_blocks.size());
            
            // Buffers for Jacobians
            std::vector<std::vector<double>> jacobian_buffers(parameter_blocks.size());

            std::vector<double*> parameter_blocks_ptrs;
            problem->GetParameterBlocksForResidualBlock(residual_block_id, &parameter_blocks_ptrs);

            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                BackendId pb_id(parameter_blocks[i].first);
                int param_dim = parameter_blocks[i].second->minimalDimension(); // Use minimal dimension (local parameterization)
                
                jacobian_buffers[i].resize(num_residuals * param_dim);
                jacobians[i] = jacobian_buffers[i].data();

                // parameter block type
                if (pb_id.type() == IdType::cPose || pb_id.type() == IdType::gPose) {
                    pose_indices.push_back(i);
                    obs_pose_ids.push_back(pb_id.asInteger());
                } else if (pb_id.type() == IdType::cLandmark) {
                    landmark_idx = i;
                    obs_landmark_id = pb_id.asInteger();
                } else if (pb_id.type() == IdType::ImuStates) {
                    speed_bias_indices.push_back(i);
                    obs_speed_bias_ids.push_back(pb_id.asInteger());
                    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                        graph->printParameterBlockInfo(parameter_blocks[i].first); 
                    }
                } else {
                    if (!problem->IsParameterBlockConstant(parameter_blocks_ptrs[i])){
                        LOG(ERROR) << "Unexpected parameter block type in visual residual: " << idTypeToString(pb_id.type());
                    }
                    jacobians[i] = nullptr;     
                }
                
                if (problem->IsParameterBlockConstant(parameter_blocks_ptrs[i])) {
                    jacobians[i] = nullptr; 
                }
            }

            // Must have at least Pose to be relevant (relaxed from Pose+Landmark to allow SpeedBias errors)
            if (pose_indices.empty() && speed_bias_indices.empty()) continue;

            if (!problem->EvaluateResidualBlock(residual_block_id, false, nullptr, residuals_eval.data(), jacobians.data())) {
                continue;
            }

            ParamResidualInfo info;
            info.residual = Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals);
            info.landmark_id = obs_landmark_id; 
            info.pose_ids = obs_pose_ids;
            info.speed_bias_ids = obs_speed_bias_ids;
            info.sig2 = 1.0; 
            info.is_current_frame = false;
            for(auto pid : obs_pose_ids) {
                if(pid == current_pose_id) {
                    info.is_current_frame = true;
                    break;
                }
            }
            info.has_landmark_jacobian = false;

            for(size_t k=0; k<pose_indices.size(); ++k) {
                int idx = pose_indices[k];
                if (jacobians[idx] != nullptr) {
                    int dim = parameter_blocks[idx].second->minimalDimension();
                    info.J_poses.push_back(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[idx], num_residuals, dim));
                } else {
                    info.J_poses.push_back(Eigen::MatrixXd()); // Empty matrix placeholder
                }
            }

            if (landmark_idx != -1 && jacobians[landmark_idx] != nullptr) {
                int dim = parameter_blocks[landmark_idx].second->minimalDimension();
                info.J_landmark = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[landmark_idx], num_residuals, dim);
                info.has_landmark_jacobian = true;
            }

            for(size_t k=0; k<speed_bias_indices.size(); ++k) {
                int idx = speed_bias_indices[k];
                if (jacobians[idx] != nullptr) {
                    int dim = parameter_blocks[idx].second->minimalDimension();
                    info.J_speed_biases.push_back(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[idx], num_residuals, dim));
                } else {
                    info.J_speed_biases.push_back(Eigen::MatrixXd()); // Empty matrix placeholder
                }
            }

            if (obs_landmark_id == 0 && num_residuals != 2 && error_type == ErrorType::kIMUError){
                info.landmark_id = reinterpret_cast<uint64_t>(residual_block_id);
            }

            all_residuals.push_back(info);
            // // print info
            // int num_param_blocks = cost_function->parameter_block_sizes().size();
            // const std::vector<int32_t>& parameter_sizes = cost_function->parameter_block_sizes();
            graph->printResidualBlockInfo(residual_block_id); // - type: ReprojectionError （46)
            // for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            //     BackendId ext_id(parameter_blocks[i].first);
            //     LOG(INFO) << "Parameter Block ID: " << ext_id.asInteger() 
            //             << " Type: " << idTypeToString(ext_id.type()) << " Dim: " << parameter_sizes[i];
            // //     graph->printParameterBlockInfo(parameter_blocks[i].first);
            // }
            // // Print calculated jacobians
            // for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            //     if (jacobians[i] != nullptr) {
            //         LOG(INFO) << "Jacobian[" << i << "] Shape: "<< num_residuals << "x" << parameter_blocks[i].second->minimalDimension();
            //         Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J_debug(jacobians[i], num_residuals, parameter_blocks[i].second->minimalDimension());
            //         LOG(INFO) << "Values:\n" << J_debug;
            //     }
            //     if (problem->IsParameterBlockConstant(parameter_blocks_ptrs[i])) {
            //         BackendId ext_id(parameter_blocks[i].first);
            //         LOG(INFO) << "Parameter block " << i  << " of " << idTypeToString(ext_id.type())  << " is constant, skipping jacobian computation.";
            //         continue;
            //     }
            // }
        }
    }

    // Sort all_residuals by landmark_id
    std::sort(all_residuals.begin(), all_residuals.end(), [](const ParamResidualInfo& a, const ParamResidualInfo& b) {
        return a.landmark_id < b.landmark_id;
    });
    
    // --- Build J_all ---
    // Maps for J_all
    std::map<uint64_t, int> all_pose_col_map;
    std::map<uint64_t, int> all_landmark_col_map;
    std::map<uint64_t, int> all_speed_bias_col_map;

    int N_all_rows = 0;
    for(const auto& res : all_residuals) {
        N_all_rows += res.residual.size();
        for(auto pid : res.pose_ids) if (pid != 0 && all_pose_col_map.find(pid) == all_pose_col_map.end()) all_pose_col_map[pid] = -1;
        if (res.landmark_id != 0 && all_landmark_col_map.find(res.landmark_id) == all_landmark_col_map.end()) all_landmark_col_map[res.landmark_id] = -1;
        for(auto sbid : res.speed_bias_ids) if (sbid != 0 && all_speed_bias_col_map.find(sbid) == all_speed_bias_col_map.end()) all_speed_bias_col_map[sbid] = -1;
    }

    // Assign columns
    int current_col = 0;
    for(auto& pair : all_pose_col_map) {
        pair.second = current_col;
        current_col += 6;

        BackendId pose_bid(pair.first);
        BackendId sb_bid = changeIdType(pose_bid, IdType::ImuStates);
        uint64_t sb_id = sb_bid.asInteger();

        if (all_speed_bias_col_map.find(sb_id) != all_speed_bias_col_map.end()) {
            all_speed_bias_col_map[sb_id] = current_col;
            current_col += 9;
        }
    }
    for(auto& pair : all_speed_bias_col_map) {
        if (pair.second == -1) {
            pair.second = current_col;
            current_col += 9;
        }
    }
    for(auto& pair : all_landmark_col_map) {
        pair.second = current_col;
        current_col += 3;
    }
    
    int N_all_cols = current_col;
    
    if (N_all_rows > 0) {
        J_all = Eigen::MatrixXd::Zero(N_all_rows, N_all_cols);
        r_all.resize(N_all_rows);
        sig2_all.resize(N_all_rows, N_all_cols);
        row_ids_all.resize(N_all_rows);
        col_ids_all.resize(N_all_cols);
        
        // Fill Col IDs
        for(auto const& pair : all_pose_col_map) {
            for(int k=0; k<6; ++k) col_ids_all[pair.second + k] = pair.first;
        }
        for(auto const& pair : all_speed_bias_col_map) {
            for(int k=0; k<9; ++k) col_ids_all[pair.second + k] = pair.first;
        }
        for(auto const& pair : all_landmark_col_map) {
            for(int k=0; k<3; ++k) col_ids_all[pair.second + k] = pair.first;
        }
        
        int current_row_idx = 0;
        for(size_t i=0; i<all_residuals.size(); ++i) {
            const auto& info = all_residuals[i];
            int num_res = info.residual.size();

            r_all.segment(current_row_idx, num_res) = info.residual;
            for(int k=0; k<num_res; ++k) {
                row_ids_all[current_row_idx + k] = info.landmark_id; 
            }
            sig2_all.block(current_row_idx, current_row_idx, num_res, num_res) = Eigen::MatrixXd::Identity(num_res, num_res) * info.sig2;
            
            for(size_t k=0; k<info.pose_ids.size(); ++k) {
                uint64_t pid = info.pose_ids[k];
                if(pid != 0 && info.J_poses[k].size() > 0) {
                    int col = all_pose_col_map[pid];
                    J_all.block(current_row_idx, col, num_res, 6) = info.J_poses[k];
                    if (info.is_current_frame) {
                        for(int r = 0; r < num_res; ++r) rows_curr.push_back(std::make_pair(info.landmark_id, current_row_idx + r));
                        for(int c = 0; c < 6; ++c) cols_curr.push_back(std::make_pair(pid, col + c));
                    }
                }
            }

            for(size_t k=0; k<info.speed_bias_ids.size(); ++k) {
                uint64_t sbid = info.speed_bias_ids[k];
                if(sbid != 0 && info.J_speed_biases[k].size() > 0) {
                    int col = all_speed_bias_col_map[sbid];
                    J_all.block(current_row_idx, col, num_res, 9) = info.J_speed_biases[k];
                    if (info.is_current_frame) {
                        for(int r = 0; r < num_res; ++r) rows_curr.push_back(std::make_pair(info.landmark_id, current_row_idx + r));
                        for(int c = 0; c < 9; ++c) cols_curr.push_back(std::make_pair(sbid, col + c));
                    }
                }
            }

            if(info.has_landmark_jacobian && info.landmark_id != 0) {
                int col = all_landmark_col_map[info.landmark_id];
                J_all.block(current_row_idx, col, num_res, 3) = info.J_landmark;
                if (info.is_current_frame) {
                    for(int k = 0; k < num_res; ++k) rows_curr.push_back(std::make_pair(info.landmark_id, current_row_idx + k));
                    for(int k = 0; k < 3; ++k) cols_curr.push_back(std::make_pair(info.landmark_id, col + k));
                }
            }
            current_row_idx += num_res;
        }
    }

    return true;
}


} // namespace gici
