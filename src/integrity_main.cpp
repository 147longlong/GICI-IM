/**
* @Function: Main function for integrity monitoring post-processing
*
* @Author  : Yulong Sun
* @Email   : sunyulong@sjtu.edu.cn
*
* Copyright (C) 2025, All rights reserved.
**/

#include "gici/integrity/visual_integrity.h"
#include <iostream>
#include <string>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h> // Add this

using namespace gici;

int main(int argc, char** argv) {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <path-to-option.yaml>" << std::endl;
      return -1;
    }

    // Load YAML
    std::string config_file_path = argv[1];
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_file_path);
    } catch (const YAML::Exception& e) {
        LOG(ERROR) << "Failed to load config file: " << config_file_path << ", " << e.what();
        return -1;
    }

    // Initialize glog for logging
    bool enable_logging = false;
    std::string snapshot_file;
    std::string csv_file;
    std::string nmea_file;
    
    // Parse options
    VisualIntegrityOptions options;
    if (config["logging"].IsDefined() && 
        option_tools::safeGet(config["logging"], "enable", &enable_logging) && 
        enable_logging == true) {
        YAML::Node logging_node = config["logging"];
        google::InitGoogleLogging("gici");
        int min_log_level = 0;
        if (option_tools::safeGet(
            logging_node, "min_log_level", &min_log_level)) {
        FLAGS_minloglevel = min_log_level;
        }
        option_tools::safeGet(logging_node, "log_to_stderr", &FLAGS_logtostderr);
        option_tools::safeGet(logging_node, "file_directory", &FLAGS_log_dir);
        if (FLAGS_logtostderr) FLAGS_stderrthreshold = min_log_level;
        else FLAGS_stderrthreshold = 5;
    }
    if (config["integrity"] && config["integrity"]["integrity_options"]) {
        YAML::Node opts = config["integrity"]["integrity_options"];
        if (opts["post_processing"]) {
            options.post_processing = opts["post_processing"].as<bool>();
        }
        if (opts["yaml_options"]) options.yaml_options = opts["yaml_options"].as<bool>();
        if (opts["use_segment"]) options.use_segment = opts["use_segment"].as<bool>();
        if (opts["start_timestamp"]) options.start_timestamp = opts["start_timestamp"].as<double>();
        if (opts["end_timestamp"]) options.end_timestamp = opts["end_timestamp"].as<double>();
        if (opts["snapshot_freq"]) options.snapshot_freq = opts["snapshot_freq"].as<double>();
        if (opts["snapshot_file"]) snapshot_file = opts["snapshot_file"].as<std::string>();
        if (opts["output_post_processing_csv"]) csv_file = opts["output_post_processing_csv"].as<std::string>();
        if (opts["output_post_processing_nmea"]) nmea_file = opts["output_post_processing_nmea"].as<std::string>();
    }
    if (options.yaml_options){
        LOG(INFO) << "Read integrity options from yaml file";
        // integrity_support_message
        if (config["integrity"] && config["integrity"]["integrity_support_message"].IsDefined()) {
            const YAML::Node& ism_msg_node = config["integrity"]["integrity_support_message"];

            option_tools::safeGet(ism_msg_node, "use_complex_visual_cov", &options.use_complex_visual_cov);
            option_tools::safeGet(ism_msg_node, "use_complex_gnss_cov", &options.use_complex_gnss_cov);
            option_tools::safeGet(ism_msg_node, "use_complex_imu_cov", &options.use_complex_imu_cov);
            option_tools::safeGet(ism_msg_node, "use_complex_others_cov", &options.use_complex_others_cov);
            option_tools::safeGet(ism_msg_node, "simple_visual_sigma", &options.simple_visual_sigma);
            option_tools::safeGet(ism_msg_node, "simple_gnss_sigma", &options.simple_gnss_sigma);
            option_tools::safeGet(ism_msg_node, "simple_imu_sigma", &options.simple_imu_sigma);
            option_tools::safeGet(ism_msg_node, "simple_others_sigma", &options.simple_others_sigma);

            option_tools::safeGet(ism_msg_node, "sigma_pixel", &options.sigma_pixel);
            if (!option_tools::safeGet(ism_msg_node, "visual_prior_fault_probability", &options.visual_prior_fault_probability)) {
                // Backward compatibility for existing yaml files.
                option_tools::safeGet(ism_msg_node, "prior_fault_probability", &options.visual_prior_fault_probability);
            }
            option_tools::safeGet(ism_msg_node, "visual_meas_dim", &options.visual_meas_dim);

            // Load overbounding function parameters
            option_tools::safeGet(ism_msg_node, "overbounding_func", &options.overbounding_func);
            if (ism_msg_node["overbounding_parameters"].IsDefined()) {
                const YAML::Node& params_node = ism_msg_node["overbounding_parameters"];
                options.overbounding_parameters.clear();
                for (size_t i = 0; i < params_node.size(); i++) {
                    options.overbounding_parameters.push_back(params_node[i].as<double>());
                }
            }

            // Load normal fit function parameters
            option_tools::safeGet(ism_msg_node, "normal_func", &options.normal_func);
            if (ism_msg_node["normal_parameters"].IsDefined()) {
                const YAML::Node& params_node = ism_msg_node["normal_parameters"];
                options.normal_parameters.clear();
                for (size_t i = 0; i < params_node.size(); i++) {
                    options.normal_parameters.push_back(params_node[i].as<double>());
                }
            }

            option_tools::safeGet(ism_msg_node, "gnss_sigma_ura", &options.gnss_sigma_ura);
            option_tools::safeGet(ism_msg_node, "gnss_sigma_ure", &options.gnss_sigma_ure);
            option_tools::safeGet(ism_msg_node, "gnss_b_nom", &options.gnss_b_nom);
            option_tools::safeGet(ism_msg_node, "gnss_sat_prior_fault_probability", &options.gnss_sat_prior_fault_probability);
            option_tools::safeGet(ism_msg_node, "gnss_const_prior_fault_probability", &options.gnss_const_prior_fault_probability);
            option_tools::safeGet(ism_msg_node, "imu_prior_fault_probability", &options.imu_prior_fault_probability);
            option_tools::safeGet(ism_msg_node, "user_F", &options.user_F);
            option_tools::safeGet(ism_msg_node, "user_Rr", &options.user_Rr);
            option_tools::safeGet(ism_msg_node, "user_a_sigma", &options.user_a_sigma);
            option_tools::safeGet(ism_msg_node, "user_b_sigma", &options.user_b_sigma);
            option_tools::safeGet(ism_msg_node, "doppler_c_sigma", &options.doppler_c_sigma);
            option_tools::safeGet(ism_msg_node, "rho_max", &options.rho_max);
            option_tools::safeGet(ism_msg_node, "psi_user_deg", &options.psi_user_deg);
            option_tools::safeGet(ism_msg_node, "k_mp", &options.k_mp);
            option_tools::safeGet(ism_msg_node, "tau_mp", &options.tau_mp);
        }

        // navigation_requirements
        if (config["integrity"] && config["integrity"]["navigation_requirements"].IsDefined()) {
            const YAML::Node& nav_req_node = config["integrity"]["navigation_requirements"];
            #define LOAD_NAV_REQ(opt) \
            if (!option_tools::safeGet(nav_req_node, #opt, &options.opt)) { \
            LOG(INFO) << "Unable to load integrity option " << #opt \
                    << ". Using default instead."; }

            LOAD_NAV_REQ(PHMI);
            LOAD_NAV_REQ(PHMI_La);
            LOAD_NAV_REQ(PHMI_Lo);
            LOAD_NAV_REQ(PHMI_V);
            
            LOAD_NAV_REQ(PFA);
            LOAD_NAV_REQ(PFA_La);
            LOAD_NAV_REQ(PFA_Lo);
            LOAD_NAV_REQ(PFA_V);

            LOAD_NAV_REQ(HAL);
            LOAD_NAV_REQ(LaAL);
            LOAD_NAV_REQ(LoAL);
            LOAD_NAV_REQ(VAL);

            LOAD_NAV_REQ(P_THRES);
            LOAD_NAV_REQ(Fc_THRES);
            LOAD_NAV_REQ(PL_TOL);
            #undef LOAD_NAV_REQ
        }
    }


    if (snapshot_file.empty()) {
        LOG(ERROR) << "snapshot_file not set in configuration!";
        return -1;
    }

    LOG(INFO) << "Starting Integrity Monitoring Post-Processing...";
    LOG(INFO) << "Snapshot File: " << snapshot_file;
    if (!csv_file.empty()) LOG(INFO) << "Output CSV: " << csv_file;
    if (!nmea_file.empty()) LOG(INFO) << "Update NMEA: " << nmea_file;

     // In fact, options loaded from snapshots.bin
    VisualIntegrity integrity(options);
    
    if (!csv_file.empty()) {
        integrity.setCsvOutputFile(csv_file);
    }
    if (!nmea_file.empty()) {
        integrity.setOutputFile(nmea_file);
    }

    integrity.processSnapshotsFromFile(snapshot_file);

    LOG(INFO) << "Integrity Monitoring Post-Processing Finished.";

    return 0;
}