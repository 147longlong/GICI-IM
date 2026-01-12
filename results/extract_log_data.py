import re
import os

def parse_log_file(log_file_path, output_file_path):
    print(f"Reading log file: {log_file_path}")
    
    # Regex patterns
    # Pattern for subset info: "Compute subset solution, number of subsets: 9147, measurement dimension: 427, time taken(s): 1.41902"
    subset_pattern = re.compile(r"Compute subset solution, number of subsets:\s*(\d+).*?time taken\(s\):\s*([\d\.]+)")
    
    # Pattern for PL info: "Timestamp: 1679304413.314000, XPL: 0.030800 m, YPL: 0.054731 m, VPL: 0.045130 m"
    pl_pattern = re.compile(r"Timestamp:\s*([\d\.]+),\s*XPL:\s*([\d\.]+)\s*m,\s*YPL:\s*([\d\.]+)\s*m,\s*VPL:\s*([\d\.]+)\s*m")
    
    results = []
    
    # Temporary storage for the last seen subset info
    current_subset_info = None
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Check for subset info
                subset_match = subset_pattern.search(line)
                if subset_match:
                    num_subsets = subset_match.group(1)
                    time_taken = subset_match.group(2)
                    current_subset_info = {
                        'subsets': num_subsets,
                        'time_taken': time_taken
                    }
                    continue
                
                # Check for PL info
                pl_match = pl_pattern.search(line)
                if pl_match:
                    if current_subset_info:
                        timestamp = pl_match.group(1)
                        xpl = pl_match.group(2)
                        ypl = pl_match.group(3)
                        vpl = pl_match.group(4)
                        
                        # Combine all info
                        record = {
                            'timestamp': timestamp,
                            'subsets': current_subset_info['subsets'],
                            'time_taken': current_subset_info['time_taken'],
                            'xpl': xpl,
                            'ypl': ypl,
                            'vpl': vpl
                        }
                        results.append(record)
                        
                        # Clear the subset info to ensure we don't reuse it (assuming 1-to-1 mapping)
                        current_subset_info = None
                    else:
                        print(f"Warning: Found PL info without preceding subset info at line: {line.strip()}")
                        
    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}")
        return

    # Write results to output file
    print(f"Writing {len(results)} records to {output_file_path}")
    with open(output_file_path, 'w') as f:
        # Header
        f.write("Timestamp,Subsets,TimeTaken(s),XPL(m),YPL(m),VPL(m)\n")
        
        for record in results:
            line = f"{record['timestamp']},{record['subsets']},{record['time_taken']},{record['xpl']},{record['ypl']},{record['vpl']}\n"
            f.write(line)

    print("Done.")

if __name__ == "__main__":
    log_file = "/home/syl/GICI-IM/results/gici.dell-PowerEdge-R750.dell.log.INFO.20260104-191933.1849065"
    output_file = "/home/syl/GICI-IM/results/sig2_int_raw.txt"
    parse_log_file(log_file, output_file)
