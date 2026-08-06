/**
* @Function: Visual Integrity Monitoring using MHSS
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#pragma once

#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>
#include "gici/estimate/graph.h"
#include "gici/utility/common.h"
#include "gici/gnss/gnss_common.h"
#include "gici/utility/rtklib_safe.h"

#include "gici/vision/visual_estimator_base.h"
#include "gici/estimate/pose_parameter_block.h"
#include "gici/estimate/speed_and_bias_parameter_block.h"

#include "gici/integrity/jacobian_visualization.h"

#include <omp.h>

namespace gici {

struct VisualIntegrityOptions {

    // Enable/Disable Integrity Monitoring
    bool enable = true;
    bool use_segment = false; // Whether to use segmentation for fault grouping

    bool use_complex_visual_cov = true;
    bool use_complex_gnss_cov = true;
    bool use_complex_imu_cov = true;
    bool use_complex_others_cov = true;

    double simple_visual_sigma = 1.0;
    double simple_gnss_sigma = 1.0;
    double simple_imu_sigma = 1.0;
    double simple_others_sigma = 1.0;

    // Integrtiy Support Message 
    double sigma_pixel = 1.0; // pixel sigma for visual residuals
    double visual_prior_fault_probability = 1.0e-4; // Prior probability of single visual feature fault
    int visual_meas_dim = 2; // 2D reprojection errors
    // Overbounding function parameters for q_i
    std::string overbounding_func = "none"; // Function type: "dual_exp" or "rational"
    std::vector<double> overbounding_parameters; // Parameters for overbounding function
    // Normal fit function parameters for q_i
    std::string normal_func = "none"; // Function type: "dual_exp" or "rational"
    std::vector<double> normal_parameters; // Parameters for normal fit function

    // GNSS ISM parameters (vector order: [GPS, GLO, BDS, GAL])
    std::vector<double> gnss_sigma_ura = {2.4, 9.0, 3.0, 7.5};
    std::vector<double> gnss_sigma_ure = {1.6, 6.0, 2.0, 5.0};
    std::vector<double> gnss_b_nom = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> gnss_sat_prior_fault_probability = {1.0e-8, 1.0e-5, 1.0e-4, 1.0e-4};
    std::vector<double> gnss_const_prior_fault_probability = {1.0e-8, 1.0e-5, 1.0e-4, 1.0e-4};
    double imu_prior_fault_probability = 1.0e-4;
    // GNSS covariance model parameters
    // System-specific factor for user GNSS sigma (vector order: [GPS, GLO, BDS, GAL])
    std::vector<double> user_F = {1.0, 5.0, 2.0, 1.5};
    // Pseudorange to Phase factor
    double user_Rr = 100.0;
    // Parameters for user sigma model: F^s^2 R_r^2 (a_σ^2+(b_σ^2)/sin^2⁡(θ_r^s ) )
    double user_a_sigma = 0.01;
    double user_b_sigma = 0.03;
    // Parameters for doppler user sigma model: F^s^2 σ_doppler^2
    double doppler_c_sigma = 0.30;
    // Upper limit value of spatial correlation coefficient
    double rho_max = 0.20;
    // User-defined angle for spatial correlation decay (in degrees)
    double psi_user_deg = 25.0;
    // Temporal correlation parameters
    double k_mp = 1.0;
    double tau_mp = 158.0;
    
    // Nvigation requirements
    double PHMI = 1.0e-7;
    double PHMI_La = 1.0e-7 * 0.33;
    double PHMI_Lo = 1.0e-7 * 0.34;
    double PHMI_V = 1.0e-7 * 0.33;

    double PFA = 1.0e-5;
    double PFA_La = 1.0e-5 * 0.33;
    double PFA_Lo = 1.0e-5 * 0.34;
    double PFA_V = 1.0e-5 * 0.33;

    double HAL = 2.0;
    double LaAL = 0.55;
    double LoAL = 0.50;
    double VAL = 1.40;

    // MHSS parameters
    double P_THRES = 2e-8;
    double Fc_THRES = 0.01;
    double PL_TOL = 1.0e-3;

    bool post_processing = true;
    double start_timestamp = 0;
    double end_timestamp = -1;
    bool yaml_options = true;
    double snapshot_freq = 1;
    std::string snapshot_file = "";
};

enum class GnssKind {None, CodeDD, PhaseDD, Doppler};

struct GenericResidualInfo {
    double timestamp = 0.0;
    std::pair<uint64_t, std::string> row_id; 
    Eigen::VectorXd residual;
    std::string error_type_str = "Unknown";
    double sig2_int = 1.0;
    double sig2_acc = 1.0;
    std::vector<std::pair<uint64_t, Eigen::MatrixXd>> jacobians; // ParamID, Jacobian

    // GNSS
    GnssKind gnss_kind = GnssKind::None;
    std::string prn = "";
    std::string ref_prn = "";
    double elevation = 0.0;
    double azimuth = 0.0;
    double ref_elevation = 0.0;
    double ref_azimuth = 0.0;
    double sigma_user = 0.0;
    double sigma_user_ref = 0.0;

    // IMU
    bool is_imu = false;
    Eigen::MatrixXd sig2_imu;

    // Camera
    int cur_track = 0;
    bool is_current_frame = false;
    uint64_t landmark_id = 0;

    // Others
    Eigen::MatrixXd sig2_others;
};

struct IntegritySnapshot {
    double timestamp;
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
    bool monitor(const std::deque<State>& states, size_t state_index, const Graph* graph,
                    const FramePtr& frame, const PointMap& landmarks_map,
                    const GnssMeasurement* measurement_rov = nullptr,
                    const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs = nullptr);

    // New methods for post-processing
    void saveSnapshot(const std::deque<State>& states, size_t state_index, const Graph* graph,
                    const FramePtr& frame, const PointMap& landmarks_map,
                    const GnssMeasurement* measurement_rov = nullptr,
                    const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs = nullptr);
    void saveDebugImage(const FramePtr& frame, const PointMap& landmarks_map, const std::string& identifier);
    void processSnapshotsFromFile(const std::string& filename);
    
    void setOutputFile(const std::string& filename) { output_file_ = filename; }
    void setCsvOutputFile(const std::string& filename) { csv_output_file_ = filename; }

    double getHPL() const { return HPL_; }
    double getLaPL() const { return LaPL_; }
    double getLoPL() const { return LoPL_; }
    double getVPL() const { return VPL_; }
    double getIR() const { return IR_; }

private:
    std::string output_file_;
    std::string csv_output_file_;

    void serializeSnapshot(const IntegritySnapshot& snapshot, std::ofstream& out);
    void deserializeSnapshot(IntegritySnapshot& snapshot, std::ifstream& in);

    // Helper functions for reading/writing options
    void serializeOptions(const VisualIntegrityOptions& options, std::ofstream& out);
    void deserializeOptions(VisualIntegrityOptions& options, std::ifstream& in);


    // Helper functions for code reuse
    bool prepareLinearSystem(const FramePtr& frame, 
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
                             std::vector<int>& curr_pose_J_cols
                            );

    bool computeIntegrityMetrics(const Eigen::MatrixXd& J_all,
                                 const Eigen::VectorXd& r_all,
                                 const Eigen::MatrixXd&  sig2_int,
                                 const Eigen::MatrixXd&  sig2_acc,
                                 const std::map<uint64_t, std::vector<int>>& curr_lm_to_J_rows,
                                 const std::map<uint64_t, int>& curr_lm_to_object_ids,
                                 const std::map<std::string, std::vector<int>>& curr_sat_to_J_rows,
                                 const std::map<uint64_t, std::vector<int>>& imu_to_J_rows,
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
                               const Eigen::MatrixXd& sig2_int,
                               const Eigen::MatrixXd& sig2_acc,
                               const std::vector<std::vector<int>>& subsets,
                               const std::vector<std::vector<int>>& fault_group_rows,
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
                   double& LaPL,
                   double& LoPL,
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
    double computeDualExpOverboundingSig2(std::vector<double> prm, int alpha, int beta);

    // --- Data Extraction ---

    bool extractFullLinearSystem(const std::deque<State>& states, const size_t state_index, const Graph* graph, const PointMap& landmarks_map,
                                 const std::deque<std::pair<GnssMeasurement, GnssMeasurement>>* gnss_measurement_pairs,
                                 Eigen::MatrixXd& J_all, Eigen::VectorXd& r_all, Eigen::MatrixXd& sig2_int, Eigen::MatrixXd& sig2_acc,
                                 std::vector<std::pair<uint64_t, std::string>>& row_ids_all, std::vector<std::pair<uint64_t, std::string>>& col_ids_all, std::vector<std::pair<uint64_t, double>>& pose_timestamps,
                                std::vector<std::pair<uint64_t, int>>& rows_curr, std::vector<std::pair<uint64_t, int>>& cols_curr,
                                std::map<uint64_t, std::vector<std::string>>& gnss_resid_to_prns,
                                std::map<uint64_t, std::vector<uint64_t>>& gnss_resid_to_param_ids,
                                std::map<uint64_t, std::vector<uint64_t>>& imu_resid_to_param_ids
                                );
    
    void extractLandmarkRowsCols(const FramePtr& frame, const PointMap& landmarks_map,
                                 const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                                 const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
                                 const std::vector<std::pair<uint64_t, int>>& cols_curr,
                                 std::map<uint64_t, std::vector<int>>& curr_landmark_observation_rows,
                                 std::map<uint64_t, std::vector<int>>& all_landmark_observation_rows,
                                 std::map<uint64_t, int>& curr_landmark_object_ids,
                                 std::map<uint64_t, int>& all_landmark_object_ids);

    void extractGnssRowsCols(const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                             const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
                             const std::vector<std::pair<uint64_t, int>>& cols_curr,
                             const std::map<uint64_t, std::vector<std::string>>& gnss_resid_to_prns,
                             const std::map<uint64_t, std::vector<uint64_t>>& gnss_resid_to_param_ids,
                             std::map<std::string, std::vector<int>>& curr_sat_observation_rows,
                             std::map<std::string, std::vector<int>>& all_sat_observation_rows);

    void extractImuRowsCols(const std::vector<std::pair<uint64_t, std::string>>& row_ids_all,
                            const std::vector<std::pair<uint64_t, std::string>>& col_ids_all,
                            const std::vector<std::pair<uint64_t, int>>& cols_curr,
                            const std::map<uint64_t, std::vector<uint64_t>>& imu_resid_to_param_ids,
                            std::map<uint64_t, std::vector<int>>& curr_imu_observation_rows,
                            std::map<uint64_t, std::vector<int>>& all_imu_observation_rows);
    void extractPoseRelatedRowsCols(uint64_t current_pose_id,
                                                  IdType current_pose_type,
                                                  std::vector<std::pair<uint64_t, int>>& cols_curr,
                                                  std::map<uint64_t, std::vector<int>>& pose_related_cols,
                                                  std::vector<int>& curr_pose_J_cols);
                    
    double computeConditionNumber(const Eigen::MatrixXd& A);

    int getGnssSystemIndex(const std::string& prn) const;
    double getBoundedProbabilityFromVector(const std::vector<double>& probs, int idx, double fallback) const;
    void addFaultGroup(const std::vector<int>& rows_in, const double p_in, const std::string& source_id,
                       std::vector<std::vector<int>>& rows_groups, std::vector<double>& p_groups,
                       std::vector<std::string>& source_ids) const;

    double getGnssSystemFactor(const std::string& prn) const;
    double computeGnssUserSigma(const double elevation, const std::string& prn) const;
    double computeGnssDopplerSigma(const std::string& prn) const;
    double computeGnssSpatialCorrelation(double az1, double el1, double az2, double el2) const;
    double computeGnssCodeDdVariance(double sigma_user, double sigma_ref, double rho_sr) const;
    bool updateGnssSatelliteInfo(const GnssMeasurement& measurement,
                                 const std::string& prn,
                                 const std::string& ref_prn,
                                 std::string& out_prn,
                                 std::string& out_ref_prn,
                                 double& out_elevation,
                                 double& out_azimuth,
                                 double& out_ref_elevation,
                                 double& out_ref_azimuth) const;

    // Robust Cholesky decomposition with multiple fallback strategies
    bool computeRobustCholesky(const Eigen::MatrixXd& A, Eigen::LLT<Eigen::MatrixXd>& llt_out, double& used_damping);
    
    // Robust weight matrix computation with validation
    bool computeRobustWeightMatrix(Eigen::MatrixXd& sig2_int, Eigen::MatrixXd& W, bool& diag_force);

    Eigen::MatrixXd robustInverse(const Eigen::MatrixXd& M, double svd_threshold = -1.0, bool always_pseudo = false);

    Eigen::MatrixXd pseudoinverseSVD(const Eigen::MatrixXd& M, double threshold = -1.0);
    
private:
    VisualIntegrityOptions options_;

    double timestamp_;
    double last_timestamp_ = 0;
    int consecutive_gpose_saved_ = 0;
    
    double HPL_;
    double LaPL_;
    double LoPL_;
    double VPL_;
    double IR_;

    bool is_first_ = true;


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

    std::vector<std::string> fault_group_source_ids_;
};

} // namespace gici
