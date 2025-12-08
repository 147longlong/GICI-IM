/**
* @Function: Visual Integrity Monitoring using MHSS
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#include "gici/integrity/visual_integrity.h"
#include "gici/integrity/jacobian_visualization.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <glog/logging.h>
#include <iomanip>
#include "gici/estimate/pose_parameter_block.h"
#include "gici/estimate/speed_and_bias_parameter_block.h"

namespace gici {


VisualIntegrity::VisualIntegrity(const VisualIntegrityOptions& options)
    : options_(options), HPL_(0), VPL_(0), IR_(0)
{
}

VisualIntegrity::~VisualIntegrity()
{
}



bool VisualIntegrity::monitor(const FramePtr& frame, const std::deque<State>& states, const Graph* graph, const PointMap& landmarks_map, size_t state_index)
{


    State state = states[state_index];
    timestamp_ = state.timestamp;
    if (!const_cast<State&>(state).valid() || state.id.type() != IdType::cPose) return false;


    Eigen::MatrixXd J_all;
    Eigen::VectorXd r_all;
    Eigen::VectorXd sig2_all;

    std::vector<std::pair<uint64_t, std::string>> row_ids_all;
    std::vector<std::pair<uint64_t, std::string>> col_ids_all;
    std::vector<std::pair<uint64_t, double>> pose_timestamps;
    std::vector<std::pair<uint64_t, int>> rows_curr;
    std::vector<std::pair<uint64_t, int>> cols_curr;   


    // 1. Extract Linear System (J * dx = r)
    if (!extractFullLinearSystem(frame, states, state_index, graph,
                             J_all, r_all, sig2_all, row_ids_all, col_ids_all, pose_timestamps, rows_curr, cols_curr)) {
        LOG(ERROR) << "[VisualIntegrity] Failed to extract linear system.";
        return false;
    }
    

    saveFactorGraphDot(graph, state.id.asInteger(), pose_timestamps, "/home/syl/GICI-IM/factor_graph.dot");
    
    printJacobianInfo(J_all, r_all, row_ids_all, col_ids_all, rows_curr, cols_curr, pose_timestamps, "/home/syl/GICI-IM/jacobian_visualization.txt");

    std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols;
    extractLandmarkRelatedRowsCols(landmarks_map, row_ids_all, cols_curr, curr_lm_to_J_rows, curr_lm_to_J_cols);
    std::map<uint64_t, std::vector<int>> curr_pose_to_J_cols;
    extractPoseRelatedRowsCols(state.id.asInteger(), cols_curr, curr_pose_to_J_cols); 

    
    int N_meas = curr_lm_to_J_rows.size();
    if (N_meas < 6) { // Need at least 6 measurements for 6DOF pose (actually depends on configuration)
        LOG(ERROR) << "[VisualIntegrity] Not enough measurements: " << N_meas;
        return false;
    }
    for (const auto& lm_rows : curr_lm_to_J_rows) {
        curr_lm_ids_.push_back(lm_rows.first);
    }

    // 2. Define Prior Probabilities
    // Assuming independent faults for each feature with a fixed probability
    double p_feat_fault = options_.p_feature_fault; 
    std::vector<double> p_prior(N_meas, p_feat_fault);

    // 3. Determine Subsets
    determineSubsets(p_prior, subsets_, pap_subset_, p_not_monitored_);
    CHECK_EQ(N_meas, subsets_[0].size());

    // 4. Compute Subset Solutions
    computeSubsetSolution(J_all, r_all, sig2_all, subsets_, curr_lm_to_J_rows, sigma_, bias_, sigma_ss_, bias_ss_, s1vec_, s2vec_, s3vec_, x_, chi2_);

    // 5. Filter out unmonitorable subsets
    filteroutSubsets(sigma_, bias_, sigma_ss_, bias_ss_, s1vec_, s2vec_, s3vec_, x_, chi2_, subsets_, pap_subset_, p_not_monitored_);

    // 6. Compute Test Thresholds
    T_ = computeTestThresholds(sigma_ss_, bias_ss_);

    // 7. Fault Detection (Check if test statistic > threshold)
    bool fault_detected = false;
    for (int i = 0; i < T_.rows(); ++i) {
        for (int q = 0; q < 3; ++q) { // x, y, z
            double test_stat = std::abs(x_(i, q) - x_(0, q)); // Difference from all-in-view
            if (test_stat > T_(i, q)) {
                fault_detected = true;
                LOG(ERROR) << "[VisualIntegrity] Fault detected in subset " << i << " axis " << q << std::endl;
            }
        }
    }

    // 8. Compute PL and IR
    computePL(sigma_, bias_, T_, pap_subset_, p_not_monitored_, VPL_, HPL_, XPL_, YPL_);
    IR_ = computeIR(sigma_, bias_, T_, pap_subset_, p_not_monitored_);

    // Log results
    LOG(INFO) << "[VisualIntegrity] timestamp: " << timestamp_
              << ", XPL: " << XPL_ << " m"
              << ", YPL: " << YPL_ << " m"
              << ", VPL: " << VPL_ << " m";

    if (fault_detected) {
        // TODO: Implement Fault Exclusion if needed
        return false; 
    }

    return (XPL_ < options_.XAL && YPL_ < options_.YAL && VPL_ < options_.VAL);
}


bool VisualIntegrity::extractFullLinearSystem(const FramePtr& frame, const std::deque<State>& states, size_t state_index, const Graph* graph,
                                              Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::VectorXd& sig2_all,
                                              std::vector<std::pair<uint64_t, std::string>>& row_ids_all, std::vector<std::pair<uint64_t, std::string>>& col_ids_all, std::vector<std::pair<uint64_t, double>>& pose_timestamps,
                                              std::vector<std::pair<uint64_t, int>>& rows_curr, std::vector<std::pair<uint64_t, int>>& cols_curr)
{
    
    
    uint64_t current_pose_id = states[state_index].id.asInteger();
    if (!graph->parameterBlockExists(current_pose_id)) return false;

    struct GenericResidualInfo {
        double timestamp;
        std::pair<uint64_t, std::string> row_id; 
        Eigen::VectorXd residual;
        double sig2;
        bool is_current_frame;
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
        row_id_type = kErrorToStr.at(graph->errorInterfacePtr(residual_block_id)->typeInfo());
        GenericResidualInfo info;
        info.timestamp = timestamp;
        info.row_id = {reinterpret_cast<uint64_t>(residual_block_id), row_id_type};
        info.residual = Eigen::Map<Eigen::VectorXd>(residuals_eval.data(), num_residuals);
        info.sig2 = 1.0; // Default
        info.is_current_frame = is_current;

        for (size_t i = 0; i < parameter_blocks.size(); ++i) {
            if (jacobians[i] != nullptr) {
                int dim = parameter_blocks[i].second->minimalDimension();
                Eigen::MatrixXd J = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(jacobians[i], num_residuals, dim);
                info.jacobians.push_back({parameter_blocks[i].first, J});
            }
        }
        all_residuals.push_back(info);

    }


    std::sort(all_residuals.begin(), all_residuals.end(), [](const GenericResidualInfo& a, const GenericResidualInfo& b) {
        if (a.jacobians.front().first != b.jacobians.front().first) {
            return a.jacobians.front().first < b.jacobians.front().first;
        }
        return a.row_id.second < b.row_id.second;
    });
    
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
        r_all.resize(N_all_rows);
        sig2_all.resize(N_all_rows);
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

        int current_row_idx = 0;
        for (auto& info : all_residuals) {
            int num_res = info.residual.size();
            r_all.segment(current_row_idx, num_res) = info.residual;
            
            for (int k = 0; k < num_res; ++k) {
                sig2_all(current_row_idx + k) = info.sig2;
                row_ids_all[current_row_idx + k] = info.row_id;
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
    }

    CHECK_EQ(J_all.rows(), r_all.size());
    CHECK_EQ(J_all.rows(), sig2_all.size());
    CHECK_EQ(J_all.rows(), row_ids_all.size());
    CHECK_EQ(J_all.cols(), col_ids_all.size());

    return true;
}

void VisualIntegrity::extractLandmarkRelatedRowsCols(const PointMap& landmarks_map,
                                                  const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_rows,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_cols)
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
            }
        }
    }
    CHECK_EQ(landmark_observation_rows.size(), landmark_observation_cols.size());
}


void VisualIntegrity::extractPoseRelatedRowsCols(uint64_t current_pose_id,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& pose_related_cols)
{


    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if (pb_id.type() == IdType::cPose && pb_id.asInteger() == current_pose_id) {
            curr_pose_J_cols_.push_back(cols_curr[i].second);
            pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
            if (curr_pose_J_cols_.size() == 6) break; // Early exit if all expected columns are found
            
        }
    }

    for (size_t i = 0; i < cols_curr.size(); ++i) {
        BackendId pb_id(cols_curr[i].first);
        if (pb_id.type() == IdType::ImuStates) {
            pb_id = changeIdType(pb_id, IdType::cPose);
            if (pb_id.asInteger() == current_pose_id) {
                curr_pose_J_cols_.push_back(cols_curr[i].second);
                pose_related_cols[cols_curr[i].first].push_back(cols_curr[i].second);
                if (curr_pose_J_cols_.size() == 15) break; // Early exit if all expected columns are found
            }
        } 
    }

    CHECK_EQ(pose_related_cols.size(), 2);
    CHECK_EQ(curr_pose_J_cols_.size(), 15);
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
    LOG(INFO) << "Info: The maximum simultanous faults need to monitor = " << N_fault_max;

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
                                            const Eigen::VectorXd& sig2,
                                            const std::vector<std::vector<int>>& subsets,
                                            const std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows,
                                            Eigen::MatrixXd& sigma,
                                            Eigen::MatrixXd& bias,
                                            Eigen::MatrixXd& sigma_ss,
                                            Eigen::MatrixXd& bias_ss,
                                            Eigen::MatrixXd& s1vec,
                                            Eigen::MatrixXd& s2vec,
                                            Eigen::MatrixXd& s3vec,
                                            Eigen::MatrixXd& x,
                                            Eigen::VectorXd& chi2)
{
    int N_sets = subsets.size();
    int N_rows = J.rows();
    int N_state = 3;
    
    // Initialize outputs
    sigma = Eigen::MatrixXd::Constant(N_sets, 3, INFINITY);
    bias = Eigen::MatrixXd::Constant(N_sets, 3, INFINITY);
    sigma_ss = Eigen::MatrixXd::Constant(N_sets, 3, INFINITY);
    bias_ss = Eigen::MatrixXd::Constant(N_sets, 3, INFINITY);
    s1vec = Eigen::MatrixXd::Constant(N_sets, N_rows, INFINITY);
    s2vec = Eigen::MatrixXd::Constant(N_sets, N_rows, INFINITY);
    s3vec = Eigen::MatrixXd::Constant(N_sets, N_rows, INFINITY);
    x = Eigen::MatrixXd::Zero(N_sets, N_state);
    chi2 = Eigen::VectorXd::Zero(N_sets);
    
    // Weight matrix
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(N_rows, N_rows);
    for (int i = 0; i < N_rows; ++i) W(i, i) = 1.0 / sig2(i);
    
    Eigen::MatrixXd JtWJ = J.transpose() * W * J;

    // std::ofstream debug_file("/home/syl/GICI-IM/debug_jtwj_cov.txt");
    // if (debug_file.is_open()) {
    //     debug_file << "=== JtWJ Info ===" << std::endl;
    //     debug_file << "Shape: " << JtWJ.rows() << " x " << JtWJ.cols() << std::endl;
    //     debug_file << "Matrix:\n" << JtWJ << std::endl << std::endl;

    //     Eigen::MatrixXd Cov_debug = JtWJ.inverse();
        
    //     debug_file << "=== Covariance (JtWJ^-1) Info ===" << std::endl;
    //     debug_file << "Shape: " << Cov_debug.rows() << " x " << Cov_debug.cols() << std::endl;
    //     debug_file << "Matrix:\n" << Cov_debug << std::endl;
        
    //     debug_file.close();
    //     std::cout << "[DEBUG] JtWJ and Covariance written to /home/syl/GICI-IM/debug_jtwj_cov.txt" << std::endl;
    // } else {
    //     std::cerr << "[DEBUG] Unable to open file for writing JtWJ debug info." << std::endl;
    // }
    // Compute solutions for each subset
    compute_S_coefficients(J, W, JtWJ, subsets, curr_lm_to_J_rows, residual, s1vec, s2vec, s3vec, x);
    
    // Compute sigma, bias, etc.
    Eigen::MatrixXd s1vec_2 = s1vec.array().square();
    Eigen::MatrixXd s2vec_2 = s2vec.array().square();   
    Eigen::MatrixXd s3vec_2 = s3vec.array().square();
    Eigen::MatrixXd s1vec_abs = s1vec.array().abs();
    Eigen::MatrixXd s2vec_abs = s2vec.array().abs();   
    Eigen::MatrixXd s3vec_abs = s3vec.array().abs();  

    // Assuming sig2 is used for both integrity and accuracy for now
    // And assuming zero bias
    Eigen::VectorXd nom_bias = Eigen::VectorXd::Zero(N_rows);

    sigma.col(0) = (s1vec_2 * sig2).array().sqrt();
    sigma.col(1) = (s2vec_2 * sig2).array().sqrt();
    sigma.col(2) = (s3vec_2 * sig2).array().sqrt();
    bias.col(0) = s1vec_abs * nom_bias;
    bias.col(1) = s2vec_abs * nom_bias;
    bias.col(2) = s3vec_abs * nom_bias;

    Eigen::MatrixXd delta_s1vec = s1vec - Eigen::MatrixXd::Ones(N_sets,1) * s1vec.row(0);
    Eigen::MatrixXd delta_s2vec = s2vec - Eigen::MatrixXd::Ones(N_sets,1) * s2vec.row(0);
    Eigen::MatrixXd delta_s3vec = s3vec - Eigen::MatrixXd::Ones(N_sets,1) * s3vec.row(0);
    
    Eigen::MatrixXd delta_s1vec_2 = delta_s1vec.array().square();
    Eigen::MatrixXd delta_s2vec_2 = delta_s2vec.array().square();
    Eigen::MatrixXd delta_s3vec_2 = delta_s3vec.array().square();
    Eigen::MatrixXd delta_s1vec_abs = delta_s1vec.array().abs();
    Eigen::MatrixXd delta_s2vec_abs = delta_s2vec.array().abs();   
    Eigen::MatrixXd delta_s3vec_abs = delta_s3vec.array().abs();  

    sigma_ss.col(0) = (delta_s1vec_2 * sig2).array().sqrt();
    sigma_ss.col(1) = (delta_s2vec_2 * sig2).array().sqrt();
    sigma_ss.col(2) = (delta_s3vec_2 * sig2).array().sqrt();
    bias_ss.col(0) = delta_s1vec_abs * nom_bias;
    bias_ss.col(1) = delta_s2vec_abs * nom_bias;
    bias_ss.col(2) = delta_s3vec_abs * nom_bias;

    // // Chi2 computation
    // Eigen::MatrixXi subset_eigen = Eigen::MatrixXi::Ones(subsets.size(), subsets[0].size()) * 2;
    // for(int i = 0; i < subsets.size();++i)
    // {
    //     for(int j = 0; j < subsets[0].size();++j)
    //     {
    //         subset_eigen(i,j) = subsets[i][j];
    //     }
    // }
    // Eigen::MatrixXd y_Gx2 = (residual * Eigen::MatrixXd::Ones(1,subset_eigen.rows()) - J * x.transpose()).array().square(); 
    // Eigen::MatrixXd sig_matrix = sig2 * Eigen::MatrixXd::Ones(1,subset_eigen.rows()); 
    // Eigen::MatrixXi W_subsets = subset_eigen.transpose();
    // Eigen::MatrixXd chi2_matrix = Eigen::MatrixXd::Zero(y_Gx2.rows(),y_Gx2.cols());
    
    // for(int i = 0; i < y_Gx2.rows(); ++i)
    // {
    //     for(int j = 0; j < y_Gx2.cols(); ++j)
    //     {
    //         chi2_matrix(i,j) = y_Gx2(i,j) * W_subsets(i,j) / sig_matrix(i,j);
    //     }
    // }
    // chi2 = chi2_matrix.colwise().sum().transpose();
}

void VisualIntegrity::compute_S_coefficients(const Eigen::MatrixXd& J,
                                            const Eigen::MatrixXd& W,
                                            const Eigen::MatrixXd& JtWJ,
                                            const std::vector<std::vector<int>>& subsets,
                                            const std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows,
                                            const Eigen::VectorXd& residual,
                                            Eigen::MatrixXd& s1vec,
                                            Eigen::MatrixXd& s2vec,
                                            Eigen::MatrixXd& s3vec,
                                            Eigen::MatrixXd& x)
{
    int N_sets = subsets.size();
    int N_J_rows = J.rows();
    int N_J_cols = J.cols();
    int N_state = 3;
    int N_meas_curr = subsets[0].size(); //landmarks is corresponding to col of subsets
    

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Precompute All-in-view Cholesky Decomposition
    // Add damping to the full matrix first
    Eigen::MatrixXd JtWJ_damped = JtWJ + 1e-9 * Eigen::MatrixXd::Identity(N_J_cols, N_J_cols);
    
    // Compute LLT of full matrix
    Eigen::LLT<Eigen::MatrixXd> llt_all(JtWJ_damped);
    if (llt_all.info() != Eigen::Success) {
        LOG(ERROR) << "Cholesky decomposition failed for all-in-view matrix!";
        return;
    }
    
    // Precompute J'W (N_cols x N_rows)
    Eigen::MatrixXd JtW_all = J.transpose() * W;
    
    // Precompute b_all = J'W * r
    Eigen::VectorXd b_all = JtW_all * residual;


    for (int i = 0; i < N_sets; ++i) {

        // auto subset_start_time = std::chrono::high_resolution_clock::now();
        // Eigen::MatrixXd W_sub = W;
        // int sum_i_nouse = 0;
        // for (int j = 0; j < N_meas_curr; ++j) {
        //     if (subsets[i][j] == 0) {
        //         uint64_t lm_id = curr_lm_ids_[j];
        //         std::vector<int> rows_to_zero = curr_lm_to_J_rows.at(lm_id);
        //         for (int jj : rows_to_zero) {
        //             W_sub(jj, jj) = 0.0;
        //         }
        //         sum_i_nouse++;
        //     }
        // }
        
        // Eigen::MatrixXd JtWJ_sub = J.transpose() * W_sub * J;

        // JtWJ_sub += 1e-9 * Eigen::MatrixXd::Identity(N_J_cols, N_J_cols);
        
        // Eigen::MatrixXd S_sub = Eigen::MatrixXd::Constant(N_J_cols, N_J_cols, INFINITY);

        // // Using LDLT for solving
        // // x = (J'WJ)^-1 J'W r
        // // S = (J'WJ)^-1 J'W
        // Eigen::MatrixXd Cov = JtWJ_sub.inverse(); // Or pseudo-inverse
        // Eigen::MatrixXd Sred = Cov * J.transpose() * W_sub;
        
        // // Check if solvable
        // if (N_meas_curr - sum_i_nouse >= 6) { // At least 6 measurements
        //     S_sub = Sred;
        // }
        
        // s1vec.row(i) = S_sub.row(curr_pose_J_cols_[0]);
        // s2vec.row(i) = S_sub.row(curr_pose_J_cols_[1]);
        // s3vec.row(i) = S_sub.row(curr_pose_J_cols_[2]);

       
        // Eigen::VectorXd x_full = (S_sub * residual).transpose();
        // x(i, 0) = x_full(curr_pose_J_cols_[0]);
        // x(i, 1) = x_full(curr_pose_J_cols_[1]);
        // x(i, 2) = x_full(curr_pose_J_cols_[2]);

        // Identify rows to remove based on current subset
        std::vector<int> rows_to_remove;
        rows_to_remove.reserve(N_meas_curr * 2); 

        for (int j = 0; j < N_meas_curr; ++j) {
            if (subsets[i][j] == 0) { // 0 means fault/exclude
                uint64_t lm_id = curr_lm_ids_[j];
                const std::vector<int>& rows = curr_lm_to_J_rows.at(lm_id);
                rows_to_remove.insert(rows_to_remove.end(), rows.begin(), rows.end());
            }
        }

        int k_rem = rows_to_remove.size();
        
        // Check observability roughly
        if (N_meas_curr - (k_rem / 2) < 6 && k_rem > 0) { 
            // Handle low observability if needed
        }

        // Copy the full LLT object to perform downdates on
        Eigen::LLT<Eigen::MatrixXd> llt_sub = llt_all;
        Eigen::VectorXd b_sub = b_all;

        // Perform Rank-1 Downdates for each removed measurement
        for (int r_idx : rows_to_remove) {
            double w_val = W(r_idx, r_idx);
            // Vector to downdate is sqrt(w) * J_row
            Eigen::VectorXd v = std::sqrt(w_val) * J.row(r_idx);
            
            // Rank-1 update with sigma = -1 (Downdate)
            // Note: Eigen's rankUpdate modifies the internal L factor
            llt_sub.rankUpdate(v, -1.0);
            
            // Update b vector: b_sub = b_all - J_row^T * w * r
            b_sub -= J.row(r_idx).transpose() * w_val * residual(r_idx);
        }

        // Solve for x: (L'L'^T) x = b_sub
        Eigen::VectorXd x_full = llt_sub.solve(b_sub);
        x(i, 0) = x_full(curr_pose_J_cols_[0]);
        x(i, 1) = x_full(curr_pose_J_cols_[1]);
        x(i, 2) = x_full(curr_pose_J_cols_[2]);

        // Compute S vectors
        // We need specific rows of S_sub = (J^T W_sub J)^-1 * J^T * W_sub
        // Let P_sub = (J^T W_sub J)^-1 = llt_sub.solve(I)
        // But we only need specific rows of S_sub.
        // S_sub_row_k = (k-th row of P_sub) * J^T * W_sub
        // To get k-th row of P_sub, we can solve A' * p_k = e_k (where e_k is unit vector)
        
        auto compute_S_row = [&](int row_idx_in_P) -> Eigen::VectorXd {
            // Create unit vector e_k
            Eigen::VectorXd e_k = Eigen::VectorXd::Zero(N_J_cols);
            e_k(row_idx_in_P) = 1.0;
            
            // Solve for k-th column of P_sub (which is also k-th row since symmetric)
            Eigen::VectorXd p_col = llt_sub.solve(e_k);
            
            // Result = p_col^T * JtW_all
            Eigen::VectorXd s_row = p_col.transpose() * JtW_all;
            
            // Zero out removed measurements in S matrix effectively
            for(int r_idx : rows_to_remove) {
                s_row(r_idx) = 0.0;
            }
            return s_row;
        };

        s1vec.row(i) = compute_S_row(curr_pose_J_cols_[0]);
        s2vec.row(i) = compute_S_row(curr_pose_J_cols_[1]);
        s3vec.row(i) = compute_S_row(curr_pose_J_cols_[2]);
        
        

        // // 单个子集计算耗时约0.08-0.1秒，修改后耗时0.002秒左右
        // auto subset_end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> subset_elapsed = subset_end_time - subset_start_time;
        // LOG(INFO) << "Info: Time taken to compute S coefficients for subset " << i << ": " << subset_elapsed.count() << " seconds.";
    }
    //整个耗时约500秒， 改进后大概12秒左右
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG(INFO) << "Info: Time taken to compute S coefficients for all subsets: " << elapsed.count() << " seconds.";
    LOG(INFO) << "Info: Number of subsets processed: " << N_sets;
    



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
        double sigma_min = sigma.row(i).minCoeff();
        if (sigma_min < INFINITY)
        {
            idx.push_back(i);
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
    boost::math::normal_distribution<double> normal_d(0.0, 1.0);
    
    // Allocation of PFA
    double Kfa_hor = -boost::math::quantile(normal_d, 0.25 * options_.PFA_HOR / (N_sets - 1));
    double Kfa_vert = -boost::math::quantile(normal_d, 0.5 * options_.PFA_VERT / (N_sets - 1));
    
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(N_sets, 3);
    
    T.col(0).array() = Kfa_hor * sigma_ss.col(0).array() + bias_ss.col(0).array();
    T.col(1).array() = Kfa_hor * sigma_ss.col(1).array() + bias_ss.col(1).array();
    T.col(2).array() = Kfa_vert * sigma_ss.col(2).array() + bias_ss.col(2).array();

    return T;
}

void VisualIntegrity::computePL(const Eigen::MatrixXd& sigma,
                                const Eigen::MatrixXd& bias,
                                const Eigen::MatrixXd& T,
                                const std::vector<double>& pap_subset,
                                double p_not_monitored,
                                double& VPL,
                                double& XPL,
                                double& YPL,
                                double& HPL)
{
    Eigen::Map<const Eigen::VectorXd> p_fault_const(pap_subset.data(), pap_subset.size());
    Eigen::VectorXd p_fault = p_fault_const;
    p_fault(0) = 2; //Server for IR and PL computation, because 2Q(***) +　Q(***)  

    // Allocation of PHMI
    double phmi_vert = options_.PHMI_VERT * (1.0 - p_not_monitored / options_.PHMI);
    double phmi_hor = options_.PHMI_HOR * (1.0 - p_not_monitored / options_.PHMI) / 2.0;
    
    VPL = computeVPL(sigma.col(2), bias.col(2), T.col(2), p_fault, phmi_vert);
    
    XPL = computeVPL(sigma.col(0), bias.col(0), T.col(0), p_fault, phmi_hor);
    YPL = computeVPL(sigma.col(1), bias.col(1), T.col(1), p_fault, phmi_hor);
    HPL = std::sqrt(XPL*XPL + YPL*YPL);
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
        if (sigma(i) == INFINITY)
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
        double VPL = INFINITY;
        return VPL;
    }

    Eigen::VectorXd sigma_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd bias_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd T_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd p_fault_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    
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
        if(phmi_right_low(i) == 1)
        {
            Klow(i) = -INFINITY;
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


bool VisualIntegrity::extractLinearSystem(const FramePtr& frame, const State& state, const Graph* graph, const PointMap& landmarks_map,
                                          Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::VectorXd& sig2_all,
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
        sig2_all.resize(N_all_rows);
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
                sig2_all(current_row_idx + k) = info.sig2;
                row_ids_all[current_row_idx + k] = info.landmark_id; 
            }
            
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
