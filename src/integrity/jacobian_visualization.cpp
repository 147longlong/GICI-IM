#include "gici/integrity/jacobian_visualization.h"
#include "gici/estimate/estimator_types.h"
#include "gici/estimate/error_interface.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <map>
#include <glog/logging.h>

namespace gici {

void saveFactorGraphDot(const Graph* graph, uint64_t current_pose_id, const std::vector<std::pair<uint64_t, double>>& pose_timestamps, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "digraph FactorGraph {\n";
    out << "  rankdir=LR;\n"; 
    out << "  splines=true;\n"; 
    out << "  nodesep=0.1;\n";
    out << "  ranksep=0.3;\n"; 
    out << "  bgcolor=\"white\";\n";
    
    out << "  node [fontname=\"Arial\", fontsize=10, penwidth=1.0];\n";
    out << "  edge [fontname=\"Arial\", fontsize=9, penwidth=1.0, color=\"#555555\"];\n";

    ceres::Problem* problem = graph->problem().get();
    std::vector<ceres::ResidualBlockId> residual_blocks;
    problem->GetResidualBlocks(&residual_blocks);

    // --- Data Collection ---
    std::set<uint64_t> c_poses;
    std::set<uint64_t> g_poses;
    std::map<uint64_t, uint64_t> pose_sb_map; 
    std::set<uint64_t> landmarks;
    
    struct FactorInfo {
        uint64_t id;
        ErrorType type;
        std::vector<uint64_t> connected_nodes;
        uint64_t primary_pose_id; 
        uint64_t secondary_pose_id;
    };
    std::vector<FactorInfo> factors;
    
    // Map to store IMU factors between poses: pair<p1, p2> -> factor_id
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> imu_factors_map;

    for (auto residual_block_id : residual_blocks) {
        gici::Graph::ParameterBlockCollection parameter_blocks = graph->parameters(residual_block_id);
        
        // Collect Nodes
        for (const auto& pb : parameter_blocks) {
            BackendId bid(pb.first);
            if (bid.type() == IdType::cPose) c_poses.insert(pb.first);
            else if (bid.type() == IdType::gPose) g_poses.insert(pb.first);
            else if (bid.type() == IdType::cLandmark) landmarks.insert(pb.first);
            else if (bid.type() == IdType::ImuStates) {
                BackendId cpose_bid = changeIdType(bid, IdType::cPose);
                if (graph->parameterBlockExists(cpose_bid.asInteger())) pose_sb_map[cpose_bid.asInteger()] = pb.first;
                else {
                    BackendId gpose_bid = changeIdType(bid, IdType::gPose);
                    if (graph->parameterBlockExists(gpose_bid.asInteger())) pose_sb_map[gpose_bid.asInteger()] = pb.first;
                }
            }
        }

        // Collect Factor
        auto err_ptr = graph->errorInterfacePtr(residual_block_id);
        if (!err_ptr) continue;
        
        FactorInfo info;
        info.id = reinterpret_cast<uint64_t>(residual_block_id);
        info.type = err_ptr->typeInfo();
        info.primary_pose_id = 0;
        info.secondary_pose_id = 0;

        std::vector<uint64_t> poses_in_factor;
        for (const auto& pb : parameter_blocks) {
            BackendId bid(pb.first);
            if (bid.type() == IdType::cPose || bid.type() == IdType::gPose) {
                poses_in_factor.push_back(pb.first);
            }
        }
        std::sort(poses_in_factor.begin(), poses_in_factor.end());
        if (!poses_in_factor.empty()) info.primary_pose_id = poses_in_factor[0];
        if (poses_in_factor.size() > 1) info.secondary_pose_id = poses_in_factor[1];
        
        // Store connected nodes for all factors
        for (const auto& pb : parameter_blocks) {
             info.connected_nodes.push_back(pb.first);
        }

        // IMU Factor Special Handling
        if (info.type == ErrorType::kIMUError && info.primary_pose_id != 0 && info.secondary_pose_id != 0) {
            imu_factors_map[{info.primary_pose_id, info.secondary_pose_id}] = info.id;
        }

        factors.push_back(info);
    }

    // Sort Poses
    std::vector<uint64_t> all_poses;
    for(uint64_t pid : c_poses) all_poses.push_back(pid);
    for(uint64_t pid : g_poses) all_poses.push_back(pid);

    std::sort(all_poses.begin(), all_poses.end(), [&](uint64_t a, uint64_t b){
        double ta = 0, tb = 0;
        for(const auto& p : pose_timestamps) if(p.first == a) { ta = p.second; break; }
        for(const auto& p : pose_timestamps) if(p.first == b) { tb = p.second; break; }
        if (std::abs(ta - tb) > 1e-6) return ta < tb;
        return a < b; 
    });

    // --- Draw Main Track (Poses + IMU Factors) ---
    out << "  // Main Track\n";
    
    for (size_t i = 0; i < all_poses.size(); ++i) {
        uint64_t pid = all_poses[i];
        
        // Draw Pose Node
        bool is_c = (c_poses.count(pid) > 0);
        bool is_curr = (pid == current_pose_id);
        std::string color = is_c ? (is_curr ? "#f74949d3" : "#E6E6FA") : (is_curr ? "#FFD700" : "#98FB98");
        std::string type_name = is_c ? "cPose" : "gPose";
        
        std::stringstream ss;
        ss << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" BGCOLOR=\"" << color << "\">";
        ss << "<TR><TD PORT=\"pose\" BORDER=\"0\">" << type_name << "</TD></TR>";
        ss << "<TR><TD PORT=\"pose_id\" BORDER=\"0\">ID:" << pid << "</TD></TR>";
        if (pose_sb_map.count(pid)) {
            ss << "<TR><TD PORT=\"sb\" BGCOLOR=\"#FFFACD\" BORDER=\"0\">SpeedBias</TD></TR>";
            ss << "<TR><TD PORT=\"sb_id\" BGCOLOR=\"#FFFACD\" BORDER=\"0\">ID:" << pose_sb_map[pid] << "</TD></TR>";
        }
        ss << "</TABLE>";

        out << "    v_" << pid << " [label=<" << ss.str() << ">, shape=plain, group=\"main_track\"];\n";

        // Check connection to next pose
        if (i + 1 < all_poses.size()) {
            uint64_t next_pid = all_poses[i+1];
            
            // Look for IMU factor
            uint64_t imu_id = 0;
            if (imu_factors_map.count({pid, next_pid})) imu_id = imu_factors_map[{pid, next_pid}];
            else if (imu_factors_map.count({next_pid, pid})) imu_id = imu_factors_map[{next_pid, pid}];

            if (imu_id != 0) {
                // Draw IMU Factor Node HERE
                std::string f_node = "f_" + std::to_string(imu_id);
                out << "    " << f_node << " [label=\"\", tooltip=\"ID:" << imu_id << "\", shape=square, style=filled, fillcolor=\"#FF6347\", width=0.2, height=0.2, group=\"main_track\"];\n";
             
                // Edges: Pose -> IMU -> NextPose
                out << "    v_" << pid << " -> " << f_node << " [weight=10, penwidth=2.0, dir=none];\n";
                out << "    " << f_node << " -> v_" << next_pid << " [weight=10, penwidth=2.0, dir=none];\n";
            } else {
                // Invisible edge to maintain straight line
                out << "    v_" << pid << " -> v_" << next_pid << " [style=invis, weight=10];\n";
            }
        }
    }

    // --- Draw Landmarks ---
    out << "  // Landmarks\n";
    for (uint64_t l_id : landmarks) {
        out << "    l_" << l_id << " [label=\"\", tooltip=\"ID:" << l_id << "\", shape=circle, style=filled, fillcolor=\"#32CD32\", width=0.1, height=0.1];\n";
    }

    // --- Draw Other Factors & Edges ---
    std::set<uint64_t> drawn_imu_factors;
    for(auto const& pair : imu_factors_map) drawn_imu_factors.insert(pair.second);

    // Sort factors for better alignment
    std::sort(factors.begin(), factors.end(), [&](const FactorInfo& a, const FactorInfo& b){
        if (a.primary_pose_id != b.primary_pose_id) return a.primary_pose_id < b.primary_pose_id;
        return a.id < b.id;
    });

    std::map<std::string, uint64_t> last_factor_id_by_type;

    for (const auto& f : factors) {
        if (drawn_imu_factors.count(f.id)) continue; // Already drawn
        
        if (f.type == ErrorType::kReprojectionError) {
            // Connect Poses to Landmarks
            uint64_t pose_id = 0;
            uint64_t lm_id = 0;
            for (uint64_t nid : f.connected_nodes) {
                BackendId bid(nid);
                if (bid.type() == IdType::cPose || bid.type() == IdType::gPose) pose_id = nid;
                if (bid.type() == IdType::cLandmark) lm_id = nid;
            }
            
            if (pose_id != 0 && lm_id != 0) {
                // Edge
                out << "    v_" << pose_id << ":n -> l_" << lm_id << " [color=\"#228B22\", penwidth=0.5, dir=none, weight=1, minlen=3];\n";
            }
        } else {
            // Other factors (Priors, etc.)
            if (f.type == ErrorType::kIMUError) continue;

            std::string f_node = "f_" + std::to_string(f.id);
            std::string type_str = "Err";
            try { type_str = kErrorToStr.at(f.type); } catch(...) {}
            std::string label = type_str + "\\nID:" + std::to_string(f.id);

            // Color palette for different error types
            static const std::vector<std::string> kColors = {
                "#e0baffff", "#FFDFBA", "#ffffbabd",  "#ffb3bbda", "#bae1ffc0", "#E6B3E6", "#B3FFFF", "#FFB3FF", 
                "#E6FFB3", "#FFD1DC", "#B3E6E6", "#E6BEFF", "#E6B3B3", "#ffaabcfa", 
                "#E6E6B3", "#FFD8B1", "#B3B3E6", "#D3D3D3", "#C0C0C0"
            };
            static std::map<std::string, int> type_color_indices;
            if (type_color_indices.find(type_str) == type_color_indices.end()) {
                type_color_indices[type_str] = type_color_indices.size();
            }
            int color_idx = type_color_indices[type_str];
            std::string color = kColors[color_idx % kColors.size()];

            out << "    " << f_node << " [label=\"" << label << "\", shape=rect, style=filled, fillcolor=\"" << color << "\", width=0.2, height=0.05, fontsize=8, group=\"" << type_str << "\"];\n";
            
            if (last_factor_id_by_type.count(type_str)) {
                std::string last_node = "f_" + std::to_string(last_factor_id_by_type[type_str]);
                out << "    " << last_node << " -> " << f_node << " [style=invis, weight=1];\n";
            }
            last_factor_id_by_type[type_str] = f.id;

            if (f.primary_pose_id != 0) {
                out << "    { rank=same; v_" << f.primary_pose_id << "; " << f_node << "; }\n";
            }

            for (uint64_t nid : f.connected_nodes) {
                if (c_poses.count(nid) || g_poses.count(nid)) {
                     out << "    v_" << nid << ":s -> " << f_node << " [color=\"#555555\", arrowhead=none];\n";
                }
            }
        }
    }

    out << "}\n";
    out.close();
    LOG(INFO) << "[VisualIntegrity] Factor graph saved to " << filename;
}
void printJacobianInfo(const Eigen::MatrixXd& J, const Eigen::VectorXd& r,
                              const std::vector<std::pair<uint64_t, std::string>>& row_ids, const std::vector<std::pair<uint64_t, std::string>>& col_ids,
                              const std::vector<std::pair<uint64_t, int>>& rows_curr, const std::vector<std::pair<uint64_t, int>>& cols_curr, std::vector<std::pair<uint64_t, double>> pose_timestamps,
                              const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return;
 // --- 1. Generate Structure Table ---
    struct TableCol {
        double timestamp;
        std::string type;
        std::string id;
        int start;
        int width;
        std::string dim_desc; 
    };
    std::vector<TableCol> table_cols;

    // Group Columns (Parameters)
    size_t c = 0;
    while(c < col_ids.size()) {
        uint64_t id = col_ids[c].first;
        std::string type = col_ids[c].second;
        int start = c;
        double timestamp = -1.0;
        for (const auto& pt : pose_timestamps) {
            if (pt.first == id) {
                timestamp = pt.second;
                break;
            }
        }

        // Find width of current parameter block
        while(c < col_ids.size() && col_ids[c].first == id) c++;
        int width = c - start;
        
        // Try to merge Pose + SpeedBias (Frame)
        bool merged = false;
        if ((type.find("cPose") != std::string::npos) && c < col_ids.size()) {
             uint64_t next_id = col_ids[c].first;
             std::string next_type = col_ids[c].second;
             
             int next_start = c;
             size_t temp_c = c;
             while(temp_c < col_ids.size() && col_ids[temp_c].first == next_id) temp_c++;
             int next_width = temp_c - next_start;


             
             if (next_type.find("ImuStates") != std::string::npos) {
                 // Merge Pose and IMU States
                 width += next_width;
                 c = temp_c;
                 table_cols.push_back({timestamp, "Frame-cPose", std::to_string(id), start, width, "(6+9)"});
                 merged = true;
             }
        }
        if (!merged && (type.find("gPose") != std::string::npos) && c < col_ids.size()) {
             uint64_t next_id = col_ids[c].first;
             std::string next_type = col_ids[c].second;
             
             int next_start = c;
             size_t temp_c = c;
             while(temp_c < col_ids.size() && col_ids[temp_c].first == next_id) temp_c++;
             int next_width = temp_c - next_start;
             
             if (next_type.find("ImuStates") != std::string::npos) {
                 // Merge Pose and IMU States
                 width += next_width;
                 c = temp_c;
                 table_cols.push_back({timestamp, "Frame-gPose", std::to_string(id), start, width, "(6+9)" });
                 merged = true;
             }
        }
        
        if (!merged) {
            if (type.find("Landmark") != std::string::npos) {
                // Merge consecutive landmarks into one column group
                if (!table_cols.empty() && table_cols.back().type == "Landmarks") {
                    table_cols.back().width += width;
                } else {
                    table_cols.push_back({ -1.0, "Landmarks", "", start, width, "(3n)" });
                }
            } else {
                std::string display_name = type;
                if (display_name.find("Pose") != std::string::npos) display_name = "Pose";
                else if (display_name.find("ImuStates") != std::string::npos) display_name = "SpeedBias";
                
                table_cols.push_back({ -1.0, display_name, std::to_string(id), start, width, "(" + std::to_string(width) + ")" });
            }
        }
    }

    struct TableRow {
        std::string name;
        int start;
        int height;
        int count; 
        int block_dim;
    };
    std::vector<TableRow> table_rows;

    // Group Rows (Residuals)
    size_t r_idx = 0;
    while(r_idx < row_ids.size()) {
        uint64_t id = row_ids[r_idx].first;
        std::string type = row_ids[r_idx].second;
        int start = static_cast<int>(r_idx);
        
        // Find height of current residual block
        while(r_idx < row_ids.size() && row_ids[r_idx].first == id) r_idx++;
        int height = static_cast<int>(r_idx) - start;
        
        // Grouping Logic
        if (type.find("Reprojection") != std::string::npos) {
            // Helper to find associated Pose ID
            auto get_pose_id = [&](int r_start, int r_h) -> std::string {
                for(const auto& col : table_cols) {
                    if (col.type.find("Pose") != std::string::npos) {
                        if (r_start + r_h <= J.rows() && col.start + col.width <= J.cols()) {
                             if (J.block(r_start, col.start, r_h, col.width).cwiseAbs().maxCoeff() > 1e-9) return col.id;
                        }
                    }
                }
                return "";
            };

            std::string curr_pose = get_pose_id(start, height);
            int count = 1;

            while(r_idx < row_ids.size()) {
                uint64_t next_id = row_ids[r_idx].first;
                std::string next_type = row_ids[r_idx].second;
                if (next_type.find("Reprojection") == std::string::npos) break;
                
                size_t temp_r = r_idx;
                while(temp_r < row_ids.size() && row_ids[temp_r].first == next_id) temp_r++;
                int next_height = static_cast<int>(temp_r) - static_cast<int>(r_idx);

                std::string next_pose = get_pose_id(static_cast<int>(r_idx), next_height);
                if (next_pose != curr_pose) break;
                
                r_idx = temp_r;
                count++;
            }
            int total_height = static_cast<int>(r_idx) - start;
            std::string name = "ReprojectionError";
            if (!curr_pose.empty()) name += "-Pose" + curr_pose;
            table_rows.push_back({ name, start, total_height, count, height }); 
        } else if (type.find("IMUError") != std::string::npos) {
            // Keep IMU Errors distinct (do not group consecutive ones)
            table_rows.push_back({ "IMU Error", start, height, 1, height });
        } else {
            // Group other identical types (e.g. Priors)
            int count = 1;
            while(r_idx < row_ids.size()) {
                 uint64_t next_id = row_ids[r_idx].first;
                 std::string next_type = row_ids[r_idx].second;
                 if (next_type != type) break;
                 
                 int next_start = static_cast<int>(r_idx);
                 size_t temp_r = r_idx;
                 while(temp_r < row_ids.size() && row_ids[temp_r].first == next_id) temp_r++;
                 int next_height = static_cast<int>(temp_r) - next_start;
                 
                 if (next_height != height) break; 
                 
                 r_idx = temp_r;
                 count++;
            }
            table_rows.push_back({ type, start, static_cast<int>(r_idx) - start, count, height });
        }
    }


    // Print Table
    out << "Jacobian Structure Table:\n";
    out << std::string(150, '=') << "\n";
    
    // Header Row 1: Type
    out << std::left << std::setw(35) << "Residual Type (Count x Dim)";
    for (const auto& col : table_cols) {
        std::string header = col.type;
        out << std::setw(21) << header;
    }
    out << "\n";

    // Header Row 2: ID
    out << std::left << std::setw(35) << "";
    for (const auto& col : table_cols) {
        std::string header = col.id;
        out << std::setw(21) << header;
    }
    out << "\n";
    
    // Header Row 3: Dimensions
    out << std::left << std::setw(35) << "";
    for (const auto& col : table_cols) {
        out << std::setw(21) << col.dim_desc;
    }
    out << "\n";

    out << std::left << std::setw(35) << "";
    for (const auto& col : table_cols) {
        std::string ts_str = (col.timestamp >= 0.0) ? std::to_string(col.timestamp) : "-";
        out << std::setw(21) << ts_str;
    }
    out << "\n" << std::string(150, '-') << "\n";

    // Rows
    for (const auto& row : table_rows) {
        std::stringstream ss_name;
        ss_name << row.name << "(" << row.count << "x" << row.block_dim << ")";
        out << std::left << std::setw(35) << ss_name.str();

        for (const auto& col : table_cols) {
            std::stringstream ss_cell;
            bool is_merged_frame = (col.width == 15 && col.type.find("Frame") != std::string::npos);
            
            if (is_merged_frame) {
                double max_pose = 0.0;
                double max_sb = 0.0;
                
                if (row.start + row.height <= J.rows() && col.start + 6 <= J.cols())
                    max_pose = J.block(row.start, col.start, row.height, 6).cwiseAbs().maxCoeff();
                
                if (row.start + row.height <= J.rows() && col.start + 15 <= J.cols())
                    max_sb = J.block(row.start, col.start + 6, row.height, 9).cwiseAbs().maxCoeff();
                
                bool has_pose = max_pose > 1e-9;
                bool has_sb = max_sb > 1e-9;
                
                if (!has_pose && !has_sb) {
                    ss_cell << "-";
                } else {
                    std::string dim_str;
                    if (has_pose && !has_sb) dim_str = "6";
                    else if (!has_pose && has_sb) dim_str = "9";
                    else dim_str = "(6+9)";
                    
                    if (row.count > 1) {
                        ss_cell << "[" << row.count << "x" << row.block_dim << "]x" << dim_str;
                    } else {
                        ss_cell << row.height << "x" << dim_str;
                    }
                }
            } else {
                double max_val = 0.0;
                if (row.start + row.height <= J.rows() && col.start + col.width <= J.cols()) {
                    max_val = J.block(row.start, col.start, row.height, col.width).cwiseAbs().maxCoeff();
                }

                if (max_val > 1e-9) {
                    if (row.count > 1) {
                        ss_cell << "[" << row.count << "x" << row.block_dim << "]x" << col.width;
                    } else {
                        ss_cell << row.height << "x" << col.width;
                    }
                } else {
                    ss_cell << "-";
                }
            }
            out << std::setw(21) << ss_cell.str();
        }
     
        out << "\n";
    }
    out << std::string(150, '=') << "\n\n";


    // Create sets for fast lookup of current frame indices
    std::set<int> curr_rows_set;
    for(const auto& p : rows_curr) curr_rows_set.insert(p.second);
    
    std::set<int> curr_cols_set;
    for(const auto& p : cols_curr) curr_cols_set.insert(p.second);

    out << "Jacobian Analysis: " << J.rows() << "x" << J.cols() << " (where * is the current frame)\n";
    out << std::left << std::setw(15) << "Row Range"
        << std::setw(15) << "Col Range"
        << std::setw(45) << "Residual(LmkID/ImuerrorID)"
        << std::setw(40) << "Parameter(ID & Type)"
        << std::setw(15) << "Block Size"
        << std::setw(15) << "Max Val" << "\n";
    out << std::string(115, '-') << "\n";

    r_idx = 0;
    while (r_idx < row_ids.size()) {
        uint64_t curr_row_id = row_ids[r_idx].first;
        size_t r_end = r_idx + 1;
        while (r_end < row_ids.size() && row_ids[r_end].first == curr_row_id) {
            r_end++;
        }
        
        size_t c_idx = 0;
        while (c_idx < col_ids.size()) {
            uint64_t curr_col_id = col_ids[c_idx].first;
            size_t c_end = c_idx + 1;
            while (c_end < col_ids.size() && col_ids[c_end].first == curr_col_id) {
                c_end++;
            }
            
            double max_val = 0.0;
            bool has_val = false;
            for (size_t i = r_idx; i < r_end; ++i) {
                for (size_t j = c_idx; j < c_end; ++j) {
                    double val = std::abs(J(i, j));
                    if (val > max_val) max_val = val;
                    if (val > 1e-9) has_val = true;
                }
            }
            
            if (has_val) {
                std::stringstream ss_row, ss_col, ss_size;
                ss_row << "[" << r_idx << "," << (r_end - 1) << "]";
                ss_col << "[" << c_idx << "," << (c_end - 1) << "]";
                ss_size << (r_end - r_idx) << "x" << (c_end - c_idx);
                
                std::string param_info = std::to_string(col_ids[c_idx].first) + " (" + col_ids[c_idx].second + ")";
                if (curr_cols_set.count(c_idx)) param_info += "*";

                std::string residual_info = std::to_string(row_ids[r_idx].first) + " (" + row_ids[r_idx].second + ")";
                if (curr_rows_set.count(r_idx)) residual_info += "*";

                out << std::left << std::setw(15) << ss_row.str()
                    << std::setw(15) << ss_col.str()
                    << std::setw(45) << residual_info
                    << std::setw(40) << param_info
                    << std::setw(15) << ss_size.str()
                    << std::setw(15) << max_val << "\n";
            }
            
            c_idx = c_end;
        }
        
        r_idx = r_end;
    }

    out << " ================================================================= " << "\n";

    out << "--- All Frames System ---\n";
    out << "Residuals: " << r.size() << "\n" << r << "\n\n";
    out << "Jacobian: " << J.rows() << "x" << J.cols() << "\n" << J << "\n\n";
    out << "Row IDs (Residual IDs): " << row_ids.size() << "\n";
    for (const auto& id : row_ids) out << id.first << " ";
    out << "\n\n";
    out << "Col IDs (Parameter IDs): " << col_ids.size() << "\n";
    for (const auto& id : col_ids) out << id.first << " ";
    out << "\n\n";

    out.close();
}

} // namespace gici
