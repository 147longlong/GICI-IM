#!/usr/bin/env python3
"""
Check whether jacobian_visualization contains all observation rows from
three *_jacobian_shape files at the same timestamp.

Configuration is embedded below; no CLI arguments are required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


CONFIG = {
    "jacobian_visualization_file": "/home/syl/GICI-IM/results/debug/jacobian_visualization1679304420.900000.txt",
    "observation_shape_files": [
        # "/home/syl/GICI-IM/results/debug/landmark_jacobian_shape_1679304420.900000.txt",
        "/home/syl/GICI-IM/results/debug/imu_jacobian_shape_1679304420.900000.txt",
        "/home/syl/GICI-IM/results/debug/gnss_sat_jacobian_shape_1679304420.900000.txt",
    ],
    # If True, also print rows that exist in jacobian_visualization but do not
    # appear in any observation shape file.
    "report_extra_rows_in_jacobian": True,
}


ROW_RANGE_RE = re.compile(r"^\[(\d+),(\d+)\]\s+\[\d+,\d+\]\s+.*\(([^)]+)\)")
ROW_INDICES_RE = re.compile(r"^\s*row_indices:\s*\[(.*)\]\s*$")
TIMESTAMP_RE = re.compile(r"^\s*timestamp:\s*([0-9]+(?:\.[0-9]+)?)\s*$")


@dataclass
class JacobianRowInfo:
    row: int
    residual_types: Set[str]


def parse_timestamp_from_shape_file(file_path: Path) -> float | None:
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TIMESTAMP_RE.match(line)
        if m:
            return float(m.group(1))
    return None


def parse_observation_rows(file_path: Path) -> Set[int]:
    rows: Set[int] = set()
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ROW_INDICES_RE.match(line)
        if not m:
            continue
        payload = m.group(1).strip()
        if not payload:
            continue
        for token in payload.split(","):
            token = token.strip()
            if token:
                rows.add(int(token))
    return rows


def parse_jacobian_rows_and_types(file_path: Path) -> Dict[int, JacobianRowInfo]:
    row_to_info: Dict[int, JacobianRowInfo] = {}

    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ROW_RANGE_RE.match(line)
        if not m:
            continue

        row_start = int(m.group(1))
        row_end = int(m.group(2))
        residual_type = m.group(3).strip()

        for r in range(row_start, row_end + 1):
            if r not in row_to_info:
                row_to_info[r] = JacobianRowInfo(row=r, residual_types=set())
            row_to_info[r].residual_types.add(residual_type)

    return row_to_info


def main() -> None:
    jac_path = Path(CONFIG["jacobian_visualization_file"]).expanduser()
    obs_paths = [Path(p).expanduser() for p in CONFIG["observation_shape_files"]]

    if not jac_path.exists():
        raise FileNotFoundError(f"jacobian_visualization file not found: {jac_path}")
    for p in obs_paths:
        if not p.exists():
            raise FileNotFoundError(f"observation shape file not found: {p}")

    # Parse timestamps from observation files to ensure they are aligned.
    ts_values: List[Tuple[Path, float | None]] = [
        (p, parse_timestamp_from_shape_file(p)) for p in obs_paths
    ]

    print("=== Timestamp Check (observation files) ===")
    valid_ts: List[float] = []
    for p, ts in ts_values:
        print(f"{p.name}: timestamp={ts}")
        if ts is not None:
            valid_ts.append(ts)

    if valid_ts:
        t_min = min(valid_ts)
        t_max = max(valid_ts)
        print(f"timestamp span: [{t_min}, {t_max}], delta={t_max - t_min}")
        if t_max - t_min > 1e-6:
            print("WARNING: observation files are not exactly the same timestamp.")
    else:
        print("WARNING: no timestamp found in observation files.")

    # Collect row indices from three observation files.
    obs_rows_union: Set[int] = set()
    per_file_counts: Dict[str, int] = {}
    for p in obs_paths:
        rows = parse_observation_rows(p)
        per_file_counts[p.name] = len(rows)
        obs_rows_union.update(rows)

    print("\n=== Observation Rows Summary ===")
    for name, cnt in per_file_counts.items():
        print(f"{name}: {cnt} unique rows")
    print(f"union rows across all observation files: {len(obs_rows_union)}")

    # Parse jacobian_visualization row coverage and residual types.
    jac_row_info = parse_jacobian_rows_and_types(jac_path)
    jac_rows = set(jac_row_info.keys())

    print("\n=== Jacobian Visualization Summary ===")
    print(f"{jac_path.name}: {len(jac_rows)} rows parsed from analysis table")

    missing_in_jac = sorted(obs_rows_union - jac_rows)
    extra_in_jac = sorted(jac_rows - obs_rows_union)

    print("\n=== Coverage Result ===")
    if not missing_in_jac:
        print("PASS: jacobian_visualization covers all rows from three observation files.")
    else:
        print(f"FAIL: {len(missing_in_jac)} observation rows are missing in jacobian_visualization.")
        print("Missing rows and residual type:")
        # Missing rows do not exist in jacobian table, so residual type is unknown.
        for r in missing_in_jac:
            print(f"  row {r}: Residual(Type)=UNKNOWN (not found in jacobian_visualization)")

    if CONFIG.get("report_extra_rows_in_jacobian", False):
        print("\n=== Extra Rows In Jacobian (not in observation files) ===")
        if not extra_in_jac:
            print("No extra rows in jacobian_visualization.")
        else:
            print(f"{len(extra_in_jac)} extra rows found.")
            print("Extra rows and residual type(s):")
            for r in extra_in_jac:
                types = sorted(jac_row_info[r].residual_types)
                print(f"  row {r}: Residual(Type)={','.join(types)}")


if __name__ == "__main__":
    main()
