#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "gici/estimate/graph.h"

namespace gici {

void saveFactorGraphDot(const Graph* graph, uint64_t current_pose_id, const std::vector<std::pair<uint64_t, double>>& pose_timestamps, const std::string& filename);

void printJacobianInfo(const Eigen::MatrixXd& J, const Eigen::VectorXd& r,
                       const std::vector<std::pair<uint64_t, std::string>>& row_ids, const std::vector<std::pair<uint64_t, std::string>>& col_ids,
                       const std::vector<std::pair<uint64_t, int>>& rows_curr, const std::vector<std::pair<uint64_t, int>>& cols_curr, std::vector<std::pair<uint64_t, double>> pose_timestamps,
                       const std::string& filename);

void saveEigenMatrixToFile(const Eigen::MatrixXd& Matrix_eigen, const std::string& filename);

void saveMeasDebugFile(const std::string& output_file,
                       double timestamp,
                       const std::map<uint64_t, std::vector<int>>& observation_rows,
                       const std::string& key_name,
                       const std::string& debug_name,
                       const std::map<uint64_t, int>* object_ids = nullptr);

void saveMeasDebugFile(const std::string& output_file,
                       double timestamp,
                       const std::map<std::string, std::vector<int>>& observation_rows,
                       const std::string& key_name,
                       const std::string& debug_name);

} // namespace gici
