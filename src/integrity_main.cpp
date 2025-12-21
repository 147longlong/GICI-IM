/**
* @Function: Main function for integrity monitoring post-processing
*
* @Author  :
* @Email   : 
*
* Copyright (C) 2025, All rights reserved.
**/

#include "gici/integrity/visual_integrity.h"
#include <iostream>
#include <string>
#include <glog/logging.h>

using namespace gici;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    std::string snapshot_file = "/home/syl/GICI-Dataset/2.1/integrity_snapshots.bin";
    std::string csv_file = "/home/syl/GICI-Dataset/2.1/integrity_results.csv";
    std::string nmea_file = "/home/syl/GICI-Dataset/2.1/srr_solution.txt";

    LOG(INFO) << "Starting Integrity Monitoring Post-Processing...";
    LOG(INFO) << "Snapshot File: " << snapshot_file;
    if (!csv_file.empty()) LOG(INFO) << "Output CSV: " << csv_file;
    if (!nmea_file.empty()) LOG(INFO) << "Update NMEA: " << nmea_file;

    VisualIntegrityOptions options; // Default options, or load from somewhere if needed
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
