#pragma once
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

} // namespace gici
