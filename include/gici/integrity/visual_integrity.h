/**
* @Function: Visual Integrity Monitoring using MHSS
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#pragma once

#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>
#include "gici/estimate/graph.h"
#include "gici/utility/common.h"
#include "gici/vision/visual_estimator_base.h"

namespace gici {

struct VisualIntegrityOptions {

    // Enable/Disable Integrity Monitoring
    bool enable = true;

    // Integrtiy Support Message 
    double sigma_pixel = 0.2; // Pixel noise std
    double p_feature_fault = 1.0e-4; // Probability of single feature fault
    int meas_dim = 2; // 2D reprojection errors


    // Nvigation requirements
    double PHMI = 1.0e-7;
    double PHMI_X = 1.0e-7 * 0.33; // Allocation
    double PHMI_Y = 1.0e-7 * 0.34; // Allocation
    double PHMI_V = 1.0e-7 * 0.33; // Allocation
    
    double PFA = 1.0e-5;
    double PFA_X = 1.0e-5 * 0.33;
    double PFA_Y = 1.0e-5 * 0.34;
    double PFA_V = 1.0e-5 * 0.33;

    double HAL = 2.0;
    double XAL = 1.50;
    double YAL = 0.55;
    double VAL = 1.40;

    // MHSS parameters
    double P_THRES = 2e-8;
    double Fc_THRES = 0.01;
    double PL_TOL = 1.0e-3;
    

    // Integrity options
    bool post_processing = true;
    std::string snapshot_file = "";
};

struct IntegritySnapshot {
    double timestamp;
    Eigen::MatrixXd J_all;
    Eigen::VectorXd r_all;
    Eigen::VectorXd sig2_all;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows;
    std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols;
    std::map<uint64_t, std::vector<int>> curr_pose_to_J_cols;
    std::vector<int> curr_pose_J_cols;
    VisualIntegrityOptions options;
};

class VisualIntegrity {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualIntegrity(const VisualIntegrityOptions& options);
    ~VisualIntegrity();

    /**
     * @brief Perform integrity monitoring for the current frame
     * @param frame The current frame with observations
     * @param state The current state (pose)
     * @param graph The optimization graph (to access residuals/Jacobians)
     * @param landmarks_map The map of landmarks containing observations and residual IDs
     * @return True if integrity check passes (PL < AL), False otherwise
     */
    bool monitor(const FramePtr& frame, const std::deque<State>& states, const Graph* graph, const PointMap& landmarks_map, size_t state_index);

    // New methods for post-processing
    void saveSnapshot(const FramePtr& frame, const std::deque<State>& states, const Graph* graph, const PointMap& landmarks_map, size_t state_index);
    void processSnapshotsFromFile(const std::string& filename);
    
    void setOutputFile(const std::string& filename) { output_file_ = filename; }
    void setCsvOutputFile(const std::string& filename) { csv_output_file_ = filename; }

    double getHPL() const { return HPL_; }
    double getXPL() const { return XPL_; }
    double getYPL() const { return YPL_; }
    double getVPL() const { return VPL_; }
    double getIR() const { return IR_; }

private:
    std::string output_file_;
    std::string csv_output_file_;

    void serializeSnapshot(const IntegritySnapshot& snapshot, std::ofstream& out);
    void deserializeSnapshot(IntegritySnapshot& snapshot, std::ifstream& in);

    // Helper functions for code reuse
    bool prepareLinearSystem(const FramePtr& frame, 
                             const std::deque<State>& states, 
                             size_t state_index, 
                             const Graph* graph, 
                             const PointMap& landmarks_map,
                             Eigen::MatrixXd& J_all, 
                             Eigen::VectorXd& r_all, 
                             Eigen::VectorXd& sig2_all,
                             std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                             std::map<uint64_t, std::vector<int>>& curr_lm_to_J_cols,
                             std::map<uint64_t, std::vector<int>>& curr_pose_to_J_cols,
                             std::vector<int>& curr_pose_J_cols);

    bool computeIntegrityMetrics(const Eigen::MatrixXd& J_all,
                                 const Eigen::VectorXd& r_all,
                                 const Eigen::VectorXd& sig2_all,
                                 const std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                                 const std::map<uint64_t, std::vector<int>>& curr_lm_to_J_cols,
                                 const std::vector<int>& curr_pose_J_cols);


    std::string idTypeToString(IdType type) {
    switch (type) {
        case IdType::cPose: return "cPose";
        case IdType::cLandmark: return "cLandmark";
        case IdType::ImuStates: return "ImuStates";
        case IdType::cExtrinsics: return "cExtrinsics";
        case IdType::gPosition: return "gPosition";
        case IdType::gVelocity: return "gVelocity";
        case IdType::gPose: return "gPose";
        case IdType::gClock: return "gClock";
        case IdType::gFrequency: return "gFrequency";
        case IdType::gTroposphere: return "gTroposphere";
        case IdType::gExtrinsics: return "gExtrinsics";
        case IdType::gAmbiguity: return "gAmbiguity";
        case IdType::gIonosphere: return "gIonosphere";
        case IdType::gIfb: return "gIfb";
        default: return "Unknown(" + std::to_string((int)type) + ")";
    }
    }

    // --- MHSS Core Functions (Adapted from integrity_test) ---

    void determineSubsets(const std::vector<double>& p_prior,
                          std::vector<std::vector<int>>& subsets,
                          std::vector<double>& pap_subset,
                          double& p_not_monitored);

    void computeSubsetSolution(const Eigen::MatrixXd& J,
                               const Eigen::VectorXd& residual,
                               const Eigen::VectorXd& sig2,
                               const std::vector<std::vector<int>>& subsets,
                               const std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows,
                               const std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols,
                               const std::vector<uint64_t>& curr_lm_ids,
                               const std::vector<int>& curr_pose_J_cols,
                               Eigen::MatrixXd& sigma,
                               Eigen::MatrixXd& bias,
                               Eigen::MatrixXd& sigma_ss,
                               Eigen::MatrixXd& bias_ss,
                               Eigen::MatrixXd& s1vec,
                               Eigen::MatrixXd& s2vec,
                               Eigen::MatrixXd& s3vec,
                               Eigen::MatrixXd& x,
                               Eigen::VectorXd& chi2);

    void compute_S_coefficients(const Eigen::MatrixXd& J,
                                const Eigen::MatrixXd& W,
                                const Eigen::MatrixXd& JtWJ,
                                const std::vector<std::vector<int>>& subsets,
                                const std::map<uint64_t, std::vector<int>> curr_lm_to_J_rows,
                                const std::map<uint64_t, std::vector<int>> curr_lm_to_J_cols,
                                const std::vector<uint64_t>& curr_lm_ids,
                                const std::vector<int>& curr_pose_J_cols,
                                const Eigen::VectorXd& residual,
                                Eigen::MatrixXd& s1vec,
                                Eigen::MatrixXd& s2vec,
                                Eigen::MatrixXd& s3vec,
                                Eigen::MatrixXd& x);

    std::vector<int> filteroutSubsets(Eigen::MatrixXd& sigma,
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
                                      double& p_not_monitored);

    Eigen::MatrixXd computeTestThresholds(const Eigen::MatrixXd& sigma_ss,
                                          const Eigen::MatrixXd& bias_ss);

    void computePL(const Eigen::MatrixXd& sigma,
                   const Eigen::MatrixXd& bias,
                   const Eigen::MatrixXd& T,
                   const std::vector<double>& pap_subset,
                   double p_not_monitored,
                   double& VPL,
                   double& XPL,
                   double& YPL,
                   double& HPL);

    double computeVPL(const Eigen::VectorXd& sigma,
                      const Eigen::VectorXd& bias,
                      const Eigen::VectorXd& T,
                      const Eigen::VectorXd& p_fault,
                      double phmi);

    double computeIR(const Eigen::MatrixXd& sigma,
                     const Eigen::MatrixXd& bias,
                     const Eigen::MatrixXd& T,
                     const std::vector<double>& pap_subset,
                     double p_not_monitored);

    // Helper functions
    int determineNfaultmax(const std::vector<double>& p, double P_THRES);
    std::vector<std::vector<int>> determine_k_subsets(int n, int k);
    int nchoosek(int n, int k);

    // --- Data Extraction ---
    bool extractLinearSystem(const FramePtr& frame, const State& state, const Graph* graph, const PointMap& landmarks_map,
                             Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::VectorXd& sig2_all,
                             std::vector<uint64_t>& row_ids_all, std::vector<uint64_t>& col_ids_all,
                             std::vector<std::pair<uint64_t, int>> rows_curr, std::vector<std::pair<uint64_t, int>> cols_curr);

    bool extractFullLinearSystem(const FramePtr& frame, const std::deque<State>& states, size_t state_index, const Graph* graph,
                                 Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::VectorXd& sig2_all,
                                 std::vector<std::pair<uint64_t, std::string>>& row_ids_all, std::vector<std::pair<uint64_t, std::string>>& col_ids_all, std::vector<std::pair<uint64_t, double>>& pose_timestamps,
                                std::vector<std::pair<uint64_t, int>>& rows_curr, std::vector<std::pair<uint64_t, int>>& cols_curr);
    
    void extractLandmarkRelatedRowsCols(const PointMap& landmarks_map,
                                                  const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_rows,
                                                  std::map<uint64_t, std::vector<int>>& landmark_observation_cols);
    void extractPoseRelatedRowsCols(uint64_t current_pose_id,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& pose_related_cols,
                                                  std::vector<int>& curr_pose_J_cols);

private:
    VisualIntegrityOptions options_;

    double timestamp_;
    
    double HPL_;
    double XPL_;
    double YPL_;
    double VPL_;
    double IR_;


    // Intermediate variables
    std::vector<std::vector<int>> subsets_;
    std::vector<double> pap_subset_;
    double p_not_monitored_;
    
    Eigen::MatrixXd sigma_;
    Eigen::MatrixXd bias_;
    Eigen::MatrixXd sigma_ss_;
    Eigen::MatrixXd bias_ss_;
    Eigen::MatrixXd x_;
    Eigen::VectorXd chi2_;
    Eigen::MatrixXd T_;
    
    Eigen::MatrixXd s1vec_;
    Eigen::MatrixXd s2vec_;
    Eigen::MatrixXd s3vec_;
};

} // namespace gici
