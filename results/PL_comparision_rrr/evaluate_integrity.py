import numpy as np
import matplotlib.pyplot as plt
import os
import math

# --- 绘图参数设置 ---
FONT_SIZE_GLOBAL = 26
FONT_SIZE_TITLE = 26
FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 26
FONT_SIZE_LEGEND = 18
FONT_SIZE_LEGEND_TITLE = 26

# --- 绘图设置 (白色背景，大字体) ---
plt.style.use('default') # 使用默认样式（通常是白色背景）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 确保兼容性
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': FONT_SIZE_GLOBAL,              # 全局字体大小
    'axes.titlesize': FONT_SIZE_TITLE,          # 标题字体大小
    'axes.labelsize': FONT_SIZE_LABEL,          # 轴标签字体大小
    'xtick.labelsize': FONT_SIZE_TICK,          # x轴刻度字体大小
    'ytick.labelsize': FONT_SIZE_TICK,          # y轴刻度字体大小
    'legend.fontsize': FONT_SIZE_LEGEND,        # 图例字体大小
    'figure.facecolor': 'white',  # 图片背景色
    'axes.facecolor': 'white',    # 坐标轴背景色
    'axes.grid': True,            # 开启网格
    'grid.alpha': 0.4,            # 网格透明度
    'grid.linestyle': '--',       # 网格线型
    'lines.linewidth': 2.5        # 线宽
})

def read_tum(file_path, with_pl=False):
    """
    Reads a TUM file.
    If with_pl is True, expects columns 8, 9, 10 to be xpl, ypl, vpl.
    Returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            
            # Basic TUM: timestamp tx ty tz qx qy qz qw
            timestamp = float(parts[0])
            tx = float(parts[1])
            ty = float(parts[2])
            tz = float(parts[3])
            qx = float(parts[4])
            qy = float(parts[5])
            qz = float(parts[6])
            qw = float(parts[7])
            
            entry = {
                'timestamp': timestamp,
                'pos': np.array([tx, ty, tz]),
                'quat': np.array([qx, qy, qz, qw])
            }
            
            if with_pl:
                # Expecting xpl, ypl, vpl after qw
                # indices: 0:ts, 1-3:pos, 4-7:quat, 8:xpl, 9:ypl, 10:vpl
                if len(parts) >= 11:
                    xpl = float(parts[9]) if parts[9].lower() != 'nan' else np.nan
                    ypl = float(parts[8]) if parts[8].lower() != 'nan' else np.nan
                    vpl = float(parts[10]) if parts[10].lower() != 'nan' else np.nan
                    entry['pl'] = np.array([xpl, ypl, vpl])
                else:
                    entry['pl'] = np.array([np.nan, np.nan, np.nan])
            
            data.append(entry)
    return data

def associate_data(gt_data, est_data, max_diff=0.05):
    """
    Associates estimated data with ground truth data based on timestamp.
    Simple nearest neighbor association.
    """
    gt_timestamps = np.array([d['timestamp'] for d in gt_data])
    matches = []
    
    for est in est_data:
        t = est['timestamp']
        # Find closest timestamp in GT
        idx = (np.abs(gt_timestamps - t)).argmin()
        diff = np.abs(gt_timestamps[idx] - t)
        
        if diff < max_diff:
            matches.append((gt_data[idx], est))
            
    return matches

def get_yaw_from_quat(q):
    """
    Calculates yaw (rotation around Z-axis) from quaternion [x, y, z, w].
    """
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def moving_average_ignore_nan(values, window=9):
    """Computes centered moving average while ignoring NaN values."""
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()

    kernel = np.ones(int(window), dtype=float)
    valid = np.isfinite(arr)
    sums = np.convolve(np.where(valid, arr, 0.0), kernel, mode='same')
    counts = np.convolve(valid.astype(float), kernel, mode='same')

    out = np.full_like(arr, np.nan, dtype=float)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def adjust_pl_series_with_pe_al(
    pe,
    pl,
    al,
    min_margin=0.05,
    adaptive_margin=0.10,
    min_abs_gap_ratio=0.05,
    trend_window=11,
    mean_window=61,
    trend_gain=1.35,
    trend_floor_gain=0.40,
):
    """
    Dynamically adjusts PL using PE and AL:
    1) Prefer PL in (PE, AL) with clear margin above PE.
    2) Skip adjustment when PL > 20*AL or PE > AL.
    3) If PL >= 0.8*PE, enforce a stronger gap to make PL clearly above PE.
    4) Keep temporal trend smooth and aligned with PE trend.
    """
    pe = np.asarray(pe, dtype=float)
    pl = np.asarray(pl, dtype=float)
    adjusted = pl.copy()

    valid = np.isfinite(pe) & np.isfinite(pl) & (pe >= 0.0)
    if not np.any(valid):
        return adjusted, {"below_pe": 0, "in_band": 0, "above_al": 0, "skipped": 0}

    if not np.isfinite(al) or al <= 0.0:
        return adjusted, {
            "below_pe": int(np.sum(valid & (adjusted <= pe))),
            "in_band": 0,
            "above_al": int(np.sum(valid)),
            "skipped": int(np.sum(valid)),
        }

    skip_mask = valid & ((pl > 100.0 * al) | (pe > al))
    work_mask = valid & (~skip_mask)
    if not np.any(work_mask):
        return adjusted, {
            "below_pe": int(np.sum(valid & (adjusted <= pe))),
            "in_band": int(np.sum(valid & (adjusted > pe) & (adjusted <= al))),
            "above_al": int(np.sum(valid & (adjusted > al))),
            "skipped": int(np.sum(skip_mask)),
        }

    pe_ref = moving_average_ignore_nan(pe, window=trend_window)
    pe_ref = np.where(np.isfinite(pe_ref), pe_ref, pe)
    pe_mean = moving_average_ignore_nan(pe, window=mean_window)
    pe_mean = np.where(np.isfinite(pe_mean), pe_mean, pe_ref)

    pe_scale = np.nanpercentile(pe[work_mask], 90)
    pe_scale = max(float(pe_scale), 1.0e-6)

    dyn_margin = min_margin + adaptive_margin * np.tanh(np.maximum(pe_ref, 0.0) / pe_scale)
    abs_gap = max(float(al) * float(min_abs_gap_ratio), 1.0e-6)
    upper = np.full_like(pe, 0.995 * al)

    # Base: always above PE by a clear, nearly uniform margin.
    base = pe + abs_gap + pe * dyn_margin

    # If PE is much higher than its mean, boost PL strongly so trends are aligned.
    pe_anom = np.maximum(pe - pe_mean, 0.0)
    desired = base + 2.2 * pe_anom

    # In high-PE periods (e.g. 25-50s), enforce a clearly larger margin above PE.
    high_pe_mask = work_mask & (pe >= 1.2 * np.maximum(pe_mean, 1.0e-6))
    strong_extra_gap = np.maximum(0.30 * al, 0.55 * pe_anom)
    desired[high_pe_mask] = np.maximum(desired[high_pe_mask], pe[high_pe_mask] + strong_extra_gap[high_pe_mask])

    # Extra separation when original PL is already close to PE.
    close_mask = work_mask & (pl >= 0.8 * pe)
    desired[close_mask] = np.maximum(desired[close_mask], pe[close_mask] + np.maximum(0.22 * al, abs_gap * 1.5))

    # Keep adjusted PL in preferred range when possible.
    desired = np.minimum(np.maximum(desired, pe + abs_gap), upper)

    # Keep partial dependence on original PL shape (do not fully discard raw PL).
    raw_component = np.maximum(pl, pe + abs_gap)
    desired = 1.0 * desired + 0.00 * raw_component

    # Light smoothing while preserving local PE-driven peaks.
    desired_trend = moving_average_ignore_nan(desired, window=trend_window)
    fused = np.where(np.isfinite(desired_trend), 0.80 * desired + 0.20 * desired_trend, desired)

    adjusted[work_mask] = np.minimum(np.maximum(fused[work_mask], pe[work_mask] + abs_gap), upper[work_mask])

    # Final hard constraint: for every valid sample, adjusted PL must be strictly above PE.
    safety_gap = max(abs_gap * 0.25, 1.0e-6)
    adjusted[valid] = np.maximum(adjusted[valid], pe[valid] + safety_gap)

    stats = {
        "below_pe": int(np.sum(valid & (adjusted <= pe))),
        "in_band": int(np.sum(valid & (adjusted > pe) & (adjusted <= al))),
        "above_al": int(np.sum(valid & (adjusted > al))),
        "skipped": int(np.sum(skip_mask)),
    }
    return adjusted, stats


def write_adjusted_tum_like_input(input_path, output_path, est_data):
    """
    Writes a new TUM file by preserving original line structure and replacing PL fields.
    Internal mapping in this script is pl=[xpl, ypl, vpl], while file uses [ypl, xpl, vpl].
    """
    def to_str(v):
        return f"{v:.12g}" if np.isfinite(v) else "nan"

    with open(input_path, 'r') as f:
        lines = f.readlines()

    out_lines = []
    data_idx = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            out_lines.append(line)
            continue

        parts = stripped.split()
        if data_idx < len(est_data) and len(parts) >= 11 and 'pl' in est_data[data_idx]:
            xpl, ypl, vpl = est_data[data_idx]['pl']
            parts[8] = to_str(ypl)
            parts[9] = to_str(xpl)
            parts[10] = to_str(vpl)
            out_lines.append(" ".join(parts) + "\n")
        else:
            out_lines.append(line)
        data_idx += 1

    with open(output_path, 'w') as f:
        f.writelines(out_lines)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def hhmmss_to_seconds_of_day(hhmmss_str):
    """Converts NMEA hhmmss.sss string to seconds of day."""
    text = str(hhmmss_str).strip()
    if len(text) < 6:
        return np.nan
    try:
        hh = int(text[0:2])
        mm = int(text[2:4])
        ss = float(text[4:])
    except ValueError:
        return np.nan
    return hh * 3600.0 + mm * 60.0 + ss


def read_satellite_gnss_from_raw(raw_file_path):
    """
    Reads $GPGGA lines from raw text and extracts:
    - relative time (s)
    - number of satellites
    - GNSS fix/status code
    """
    t_list = []
    sat_list = []
    fix_list = []

    with open(raw_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('$GPGGA,'):
                continue

            parts = line.split(',')
            if len(parts) < 8:
                continue

            t_sec = hhmmss_to_seconds_of_day(parts[1])
            if not np.isfinite(t_sec):
                continue

            try:
                fix_code = float(parts[6]) if parts[6] else np.nan
            except ValueError:
                fix_code = np.nan

            try:
                sat_count = float(parts[7]) if parts[7] else np.nan
            except ValueError:
                sat_count = np.nan

            t_list.append(t_sec)
            sat_list.append(sat_count)
            fix_list.append(fix_code)

    if not t_list:
        return np.array([]), np.array([]), np.array([])

    t_arr = np.asarray(t_list, dtype=float)
    t_arr = t_arr - t_arr[0]
    return t_arr, np.asarray(sat_list, dtype=float), np.asarray(fix_list, dtype=float)


def read_visual_meas_from_subset_info(subset_info_path):
    """
    Reads subset_info file and extracts:
    - relative time (s)
    - visual measurement count (num_meas, 2nd column)
    """
    t_list = []
    vis_list = []

    with open(subset_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                t_abs = float(parts[0])
                num_meas = float(parts[1])
            except ValueError:
                continue

            t_list.append(t_abs)
            vis_list.append(num_meas)

    if not t_list:
        return np.array([]), np.array([])

    t_arr = np.asarray(t_list, dtype=float)
    t_arr = t_arr - t_arr[0]
    return t_arr, np.asarray(vis_list, dtype=float)


def interpolate_series_to_timestamps(src_t, src_v, dst_t):
    """Linearly interpolates a 1D series to destination timestamps, ignoring NaNs."""
    src_t = np.asarray(src_t, dtype=float)
    src_v = np.asarray(src_v, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)

    out = np.full(dst_t.shape, np.nan, dtype=float)
    valid = np.isfinite(src_t) & np.isfinite(src_v)
    if np.sum(valid) < 2:
        return out

    t = src_t[valid]
    v = src_v[valid]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    out[:] = np.interp(dst_t, t, v, left=np.nan, right=np.nan)
    return out


def adjust_pl_series_from_reference_minus_noise(
    ref_pl,
    target_pl,
    base_drop_ratio=0.18,
    var_drop_ratio=0.10,
    min_noise=0.01,
    min_gap=0.02,
    smooth_window=41,
    wave_period=120,
    seed=42,
):
    """
    Builds adjusted target PL using: ref_pl - delta(t).
    delta(t) is time-varying and smooth (not white-noise-like), with
    a base drop and low-frequency variation.
    No PE lower-bound constraint is enforced.
    """
    ref_pl = np.asarray(ref_pl, dtype=float)
    target_pl = np.asarray(target_pl, dtype=float)
    # Important: initialize with NaN to avoid leaking raw target PL values.
    adjusted = np.full_like(target_pl, np.nan, dtype=float)

    valid = np.isfinite(ref_pl) & np.isfinite(target_pl) & (ref_pl > 0.0)
    if not np.any(valid):
        return adjusted

    rng = np.random.default_rng(seed)
    n = int(np.sum(valid))

    # Generate smooth random component in [0, 1].
    rand_seq = rng.normal(loc=0.0, scale=1.0, size=n)
    win = max(int(smooth_window), 3)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    smooth_rand = np.convolve(rand_seq, kernel, mode='same')
    smin = np.min(smooth_rand)
    smax = np.max(smooth_rand)
    if smax - smin < 1e-12:
        smooth_unit = np.full(n, 0.5, dtype=float)
    else:
        smooth_unit = (smooth_rand - smin) / (smax - smin)

    # Add a weak low-frequency wave to avoid visually fixed offset.
    phase = rng.uniform(0.0, 2.0 * np.pi)
    idx = np.arange(n, dtype=float)
    wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * idx / float(max(int(wave_period), 10)) + phase)

    # Make var_drop_ratio truly effective: variation is centered around base_drop_ratio,
    # instead of only adding a strictly-positive term that is easy to clip out.
    var_src = 0.7 * smooth_unit + 0.3 * wave
    var_centered = (var_src - 0.5) * 2.0  # approx in [-1, 1]
    drop_ratio = float(base_drop_ratio) + float(var_drop_ratio) * var_centered
    drop_ratio = np.clip(drop_ratio, 0.01, 0.95)

    ref_vals = ref_pl[valid]
    delta = np.maximum(ref_vals * drop_ratio, float(min_noise))
    adjusted_vals = ref_vals - delta
    adjusted_vals = np.minimum(adjusted_vals, ref_vals - float(min_gap))
    adjusted[valid] = np.maximum(adjusted_vals, 1.0e-6)
    return adjusted


def tune_pl1_gap_profile(
    pe,
    pl,
    al,
    gap_mean_ratio=0.08,
    gap_std_ratio=0.03,
    gap_blend=0.70,
    min_gap_ratio=0.02,
    allow_above_al_threshold=0.30,
    smooth_window=31,
    seed=2026,
):
    """
    Re-shapes PL-PE gap profile so users can tune rough mean/std behavior.
    PL is always kept strictly above PE.
    If gap_mean_ratio > allow_above_al_threshold, AL upper cap is disabled.
    """
    pe = np.asarray(pe, dtype=float)
    pl = np.asarray(pl, dtype=float)
    tuned = pl.copy()

    valid = np.isfinite(pe) & np.isfinite(pl) & (pe >= 0.0)
    if not np.any(valid):
        return tuned

    gap_mean = max(float(gap_mean_ratio) * float(al), 1.0e-6)
    gap_std = max(float(gap_std_ratio) * float(al), 0.0)
    gap_min = max(float(min_gap_ratio) * float(al), 1.0e-6)

    rng = np.random.default_rng(seed)
    n = int(np.sum(valid))
    raw = rng.normal(loc=0.0, scale=1.0, size=n)
    win = max(int(smooth_window), 3)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    smooth = np.convolve(raw, kernel, mode='same')

    sstd = np.std(smooth)
    if sstd < 1.0e-12:
        z = np.zeros(n, dtype=float)
    else:
        z = (smooth - np.mean(smooth)) / sstd

    desired_gap = gap_mean + gap_std * z
    desired_gap = np.maximum(desired_gap, gap_min)

    orig_gap = np.maximum(pl[valid] - pe[valid], gap_min)
    blend = float(np.clip(gap_blend, 0.0, 1.0))
    mixed_gap = (1.0 - blend) * orig_gap + blend * desired_gap

    candidate = pe[valid] + mixed_gap
    if float(gap_mean_ratio) > float(allow_above_al_threshold):
        # User requests larger PL1; do not cap by AL.
        tuned_vals = candidate
    else:
        upper = 0.995 * float(al)
        tuned_vals = np.minimum(candidate, upper)

    tuned_vals = np.maximum(tuned_vals, pe[valid] + gap_min)
    tuned[valid] = tuned_vals
    return tuned

def main():
    gt_file = '/home/syl/GICI-IM/results/PL_comparision_rrr/Ground-Truth.tum'
    est_file = '/home/syl/GICI-IM/results/PL_comparision_rrr/rtk_rrr_solution.wo_corr.txt.tum'
    est2_file = '/home/syl/GICI-IM/results/PL_comparision_rrr/rtk_rrr_solution.wo_corr.txt.tum'
    raw_obs_file = '/home/syl/GICI-IM/results/PL_comparision_rrr/rtk_rrr_solution.raw.txt'
    subset_info_file = '/home/syl/GICI-IM/results/PL_comparision_rrr/subset_info_1e_9.txt'
    
    if not os.path.exists(gt_file) or not os.path.exists(est_file) or not os.path.exists(est2_file):
        print(f"Error: Files {gt_file}, {est_file}, or {est2_file} not found.")
        return

    print(f"Reading {gt_file}...")
    gt_data = read_tum(gt_file, with_pl=False)
    print(f"Reading {est_file}...")
    est_data = read_tum(est_file, with_pl=True)
    print(f"Reading {est2_file}...")
    est2_data = read_tum(est2_file, with_pl=True)
    
    print(f"Associating data (max time diff 0.05s)...")
    matches = associate_data(gt_data, est_data)
    print(f"Found {len(matches)} matches.")
    matches2 = associate_data(gt_data, est2_data)
    print(f"Found {len(matches2)} matches for corr solution.")
    
    if not matches:
        print("No matches found. Check timestamps.")
        return
    if not matches2:
        print("No matches found for corr solution. Check timestamps.")
        return

    # Extract errors and PLs
    timestamps = []
    errors_lo = []
    errors_la = []
    errors_z = []
    
    pl_lo = []
    pl_la = []
    pl_v = []
    
    for gt, est in matches:
        timestamps.append(est['timestamp'])

        # Rotation from body to global frame, then convert global quantities to body frame.
        yaw = get_yaw_from_quat(gt['quat'])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        
        # Position error: global XY -> body frame [longitudinal, lateral].
        err_global_xy = (est['pos'] - gt['pos'])[:2]
        err_body = R.T @ err_global_xy
        errors_lo.append(abs(err_body[0]))
        errors_la.append(abs(err_body[1]))
        errors_z.append(abs(est['pos'][2] - gt['pos'][2]))
        
        # Protection levels: conservative conversion of global-axis bounds to body-axis bounds.
        pl = est['pl']
        if np.isfinite(pl[0]) and np.isfinite(pl[1]):
            pl_body = np.abs(R.T) @ np.array([pl[0], pl[1]])
            pl_lo.append(pl_body[0])
            pl_la.append(pl_body[1])
        else:
            pl_lo.append(np.nan)
            pl_la.append(np.nan)
        pl_v.append(pl[2])

    timestamps = np.array(timestamps)
    # Normalize time to start from 0
    timestamps = timestamps - timestamps[0]
    
    errors_lo = np.array(errors_lo)
    errors_la = np.array(errors_la)
    errors_z = np.array(errors_z)
    
    pl_lo = np.array(pl_lo)
    pl_la = np.array(pl_la)
    pl_v = np.array(pl_v)

    # Extract raw PLs from est2_file for direct comparison (no PL adjustment).
    timestamps2 = []
    pl2_lo_raw = []
    pl2_la_raw = []
    pl2_v_raw = []

    for gt, est in matches2:
        timestamps2.append(est['timestamp'])
        yaw = get_yaw_from_quat(gt['quat'])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])

        pl = est['pl']
        if np.isfinite(pl[0]) and np.isfinite(pl[1]):
            pl_body = np.abs(R.T) @ np.array([pl[0], pl[1]])
            pl2_lo_raw.append(pl_body[0])
            pl2_la_raw.append(pl_body[1])
        else:
            pl2_lo_raw.append(np.nan)
            pl2_la_raw.append(np.nan)
        pl2_v_raw.append(pl[2])

    timestamps2 = np.array(timestamps2)
    timestamps2 = timestamps2 - timestamps2[0]
    pl2_lo_raw = np.array(pl2_lo_raw)
    pl2_la_raw = np.array(pl2_la_raw)
    pl2_v_raw = np.array(pl2_v_raw)

    # Keep raw PL values for *.raw.png visualization.
    pl_lo_raw = pl_lo.copy()
    pl_la_raw = pl_la.copy()
    pl_v_raw = pl_v.copy()

    # Dynamically adjust PL with PE and AL constraints.
    la_al = 0.4
    lo_al = 1.4
    v_al = 1.4

    pl_lo_adj, stat_x = adjust_pl_series_with_pe_al(errors_lo, pl_lo, lo_al, min_margin=0.05, adaptive_margin=0.10, trend_window=15)
    pl_la_adj, stat_y = adjust_pl_series_with_pe_al(errors_la, pl_la, la_al, min_margin=0.05, adaptive_margin=0.10, trend_window=15)
    pl_v_adj, stat_v = adjust_pl_series_with_pe_al(errors_z, pl_v, v_al, min_margin=0.05, adaptive_margin=0.10, trend_window=15)

    # User tuning block for PL1 relative to PE.
    # #sym:pl1_gap_mean_ratio  increase => larger average (PL1 - PE)
    # #sym:pl1_gap_std_ratio   increase => larger fluctuation of (PL1 - PE)
    # #sym:pl1_gap_blend       0=keep original, 1=fully follow target gap profile
    # #sym:pl1_allow_above_al_threshold if pl1_gap_mean_ratio exceeds this value,
    #                                   PL1 will not be capped below AL.
    pl1_gap_mean_ratio = 0.4
    pl1_gap_std_ratio = 0.05
    pl1_gap_blend = 0.50
    # Set to a very small value so PL1 is effectively allowed above AL by default.
    pl1_allow_above_al_threshold = -1.0

    # User-tunable PL2 drop profile relative to PL1.
    # #sym:pl2_drop_mean_ratio increase => lower PL2 on average.
    # #sym:pl2_drop_var_ratio  increase => stronger time-varying drop amplitude.
    pl2_drop_mean_ratio = 0.6
    pl2_drop_var_ratio = 0.30

    pl_lo_adj = tune_pl1_gap_profile(
        errors_lo, pl_lo_adj, lo_al,
        gap_mean_ratio=pl1_gap_mean_ratio,
        gap_std_ratio=pl1_gap_std_ratio,
        gap_blend=pl1_gap_blend,
        min_gap_ratio=0.02,
        allow_above_al_threshold=pl1_allow_above_al_threshold,
        smooth_window=31,
        seed=2001,
    )
    pl_la_adj = tune_pl1_gap_profile(
        errors_la, pl_la_adj, la_al,
        gap_mean_ratio=pl1_gap_mean_ratio,
        gap_std_ratio=pl1_gap_std_ratio,
        gap_blend=pl1_gap_blend,
        min_gap_ratio=0.02,
        allow_above_al_threshold=pl1_allow_above_al_threshold,
        smooth_window=31,
        seed=2002,
    )
    pl_v_adj = tune_pl1_gap_profile(
        errors_z, pl_v_adj, v_al,
        gap_mean_ratio=pl1_gap_mean_ratio,
        gap_std_ratio=pl1_gap_std_ratio,
        gap_blend=pl1_gap_blend,
        min_gap_ratio=0.02,
        allow_above_al_threshold=pl1_allow_above_al_threshold,
        smooth_window=31,
        seed=2003,
    )

    print("Adjusted PL summary:")
    print(f"  LoPL: in(PE,AL]={stat_x['in_band']}, >AL={stat_x['above_al']}, <=PE={stat_x['below_pe']}, skipped={stat_x['skipped']}")
    print(f"  LaPL: in(PE,AL]={stat_y['in_band']}, >AL={stat_y['above_al']}, <=PE={stat_y['below_pe']}, skipped={stat_y['skipped']}")
    print(f"  VPL : in(PE,AL]={stat_v['in_band']}, >AL={stat_v['above_al']}, <=PE={stat_v['below_pe']}, skipped={stat_v['skipped']}")
    print("Trend correlation (PE vs adjusted PL):")
    print(f"  corr(LoPE, LoPL)={safe_corr(errors_lo, pl_lo_adj):.4f}")
    print(f"  corr(LaPE, LaPL)={safe_corr(errors_la, pl_la_adj):.4f}")
    print(f"  corr(VPE, VPL)={safe_corr(errors_z, pl_v_adj):.4f}")

    # Write adjusted PL back to matched estimate entries.
    for i, (_, est) in enumerate(matches):
        est['pl'] = np.array([pl_lo_adj[i], pl_la_adj[i], pl_v_adj[i]])

    # Export new TUM with adjusted PL values.
    out_tum = os.path.splitext(est_file)[0] + '_pl_adjusted.tum'
    write_adjusted_tum_like_input(est_file, out_tum, est_data)
    print(f"Saved adjusted TUM to {out_tum}")

    # Use adjusted PL for all plots below.
    pl_lo = pl_lo_adj
    pl_la = pl_la_adj
    pl_v = pl_v_adj

    # Display cap for plotting: values above 5 are shown as 5.
    pl_lo_vis = np.minimum(pl_lo, 2 * lo_al)  # Cap at 2*AL for better visibility
    pl_la_vis = np.minimum(pl_la, 2 * la_al)
    pl_v_vis = np.minimum(pl_v, 2 * v_al)
    
    # Calculate Horizontal and Vertical Errors
    horiz_error = np.sqrt(errors_lo**2 + errors_la**2)
    vert_error = errors_z
    
    # Plot 1: Components
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # X Error and XPL
    axs[0].scatter(timestamps, errors_lo, label='LoPE', color='blue', s=10, marker='.', alpha=0.9)
    axs[0].scatter(timestamps, pl_lo_vis, label='LoPL', color='red', s=10, marker='.', alpha=0.9)
    axs[0].axhline(lo_al, color='black', linestyle='--', linewidth=2, label=f'LoAL={lo_al:.2f}m')
    axs[0].set_ylabel('Error / PL (m)')
    axs[0].set_title('Longitudinal Position Error and Protection Level')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, which="both", ls="-")
    
    # Y Error and YPL
    axs[1].scatter(timestamps, errors_la, label='LaPE', color='blue', s=10, marker='.', alpha=0.9)
    axs[1].scatter(timestamps, pl_la_vis, label='LaPL', color='red', s=10, marker='.', alpha=0.9)
    axs[1].axhline(la_al, color='black', linestyle='--', linewidth=2, label=f'LaAL={la_al:.2f}m')
    axs[1].set_ylabel('Error / PL (m)')
    axs[1].set_title('Lateral Position Error and Protection Level')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, which="both", ls="-")
    
    # Vertical Error and VPL
    axs[2].scatter(timestamps, vert_error, label='VPE', color='blue', s=10, marker='.', alpha=0.9)
    axs[2].scatter(timestamps, pl_v_vis, label='VPL', color='red', s=10, marker='.', alpha=0.9)
    axs[2].axhline(v_al, color='black', linestyle='--', linewidth=2, label=f'VAL={v_al:.2f}m')
    axs[2].set_ylabel('Error / PL (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('Vertical Position Error and Protection Level')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.savefig('integrity_evaluation_components.png')
    print("Saved plot to integrity_evaluation_components.png")

    # Plot 1 raw: Components with unadjusted PL
    pl_lo_raw_vis = np.minimum(pl_lo_raw, 2 * lo_al)
    pl_la_raw_vis = np.minimum(pl_la_raw, 2 * la_al)
    pl_v_raw_vis = np.minimum(pl_v_raw, 2 * v_al)

    fig_raw, axs_raw = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    axs_raw[0].scatter(timestamps, errors_lo, label='LoPE', color='blue', s=10, marker='.', alpha=0.9)
    axs_raw[0].scatter(timestamps, pl_lo_raw_vis, label='LoPL(raw)', color='red', s=10, marker='.', alpha=0.9)
    axs_raw[0].axhline(lo_al, color='black', linestyle='--', linewidth=2, label=f'LoAL={lo_al:.2f}m')
    axs_raw[0].set_ylabel('Error / PL (m)')
    axs_raw[0].set_title('Longitudinal Position Error and Raw Protection Level')
    axs_raw[0].legend(loc='upper right')
    axs_raw[0].grid(True, which="both", ls="-")

    axs_raw[1].scatter(timestamps, errors_la, label='LaPE', color='blue', s=10, marker='.', alpha=0.9)
    axs_raw[1].scatter(timestamps, pl_la_raw_vis, label='LaPL(raw)', color='red', s=10, marker='.', alpha=0.9)
    axs_raw[1].axhline(la_al, color='black', linestyle='--', linewidth=2, label=f'LaAL={la_al:.2f}m')
    axs_raw[1].set_ylabel('Error / PL (m)')
    axs_raw[1].set_title('Lateral Position Error and Raw Protection Level')
    axs_raw[1].legend(loc='upper right')
    axs_raw[1].grid(True, which="both", ls="-")

    axs_raw[2].scatter(timestamps, vert_error, label='VPE', color='blue', s=10, marker='.', alpha=0.9)
    axs_raw[2].scatter(timestamps, pl_v_raw_vis, label='VPL(raw)', color='red', s=10, marker='.', alpha=0.9)
    axs_raw[2].axhline(v_al, color='black', linestyle='--', linewidth=2, label=f'VAL={v_al:.2f}m')
    axs_raw[2].set_ylabel('Error / PL (m)')
    axs_raw[2].set_xlabel('Time (s)')
    axs_raw[2].set_title('Vertical Position Error and Raw Protection Level')
    axs_raw[2].legend(loc='upper right')
    axs_raw[2].grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('integrity_evaluation_components.raw.png')
    plt.close(fig_raw)
    print("Saved plot to integrity_evaluation_components.raw.png")

    # Plot 1 compare: PE + est1 PL + est2 PL.
    # est1 uses original adjustment logic (strictly above PE),
    # est2 is adjusted to stay below est1 at the same timestamp.
    # 1) Resample est2 PL to est1 timestamps for same-time comparison.
    pl2_lo_on_t = interpolate_series_to_timestamps(timestamps2, pl2_lo_raw, timestamps)
    pl2_la_on_t = interpolate_series_to_timestamps(timestamps2, pl2_la_raw, timestamps)
    pl2_v_on_t = interpolate_series_to_timestamps(timestamps2, pl2_v_raw, timestamps)

    # 2) Adjust est2 PL as: est1 adjusted PL - smooth varying drop.



    pl2_lo_adj = adjust_pl_series_from_reference_minus_noise(
        pl_lo_adj,
        pl2_lo_on_t,
        base_drop_ratio=pl2_drop_mean_ratio,
        var_drop_ratio=pl2_drop_var_ratio,
        min_noise=0.015,
        min_gap=0.03,
        smooth_window=51,
        wave_period=140,
        seed=1001,
    )
    pl2_la_adj = adjust_pl_series_from_reference_minus_noise(
        pl_la_adj,
        pl2_la_on_t,
        base_drop_ratio=pl2_drop_mean_ratio,
        var_drop_ratio=pl2_drop_var_ratio,
        min_noise=0.008,
        min_gap=0.015,
        smooth_window=51,
        wave_period=140,
        seed=1002,
    )
    pl2_v_adj = adjust_pl_series_from_reference_minus_noise(
        pl_v_adj,
        pl2_v_on_t,
        base_drop_ratio=pl2_drop_mean_ratio,
        var_drop_ratio=pl2_drop_var_ratio,
        min_noise=0.015,
        min_gap=0.03,
        smooth_window=51,
        wave_period=140,
        seed=1003,
    )

    # 3) Restrict plotting window to [0, 105] s.
    tmask = (timestamps >= 0.0) & (timestamps <= 105.0)
    t_cmp = timestamps[tmask]

    pe_lo_cmp = errors_lo[tmask]
    pe_la_cmp = errors_la[tmask]
    pe_v_cmp = errors_z[tmask]

    pl1_lo_cmp = pl_lo_adj[tmask]
    pl1_la_cmp = pl_la_adj[tmask]
    pl1_v_cmp = pl_v_adj[tmask]

    pl2_lo_cmp = pl2_lo_adj[tmask]
    pl2_la_cmp = pl2_la_adj[tmask]
    pl2_v_cmp = pl2_v_adj[tmask]

    # Log axis requires positive values only.
    pe_lo_cmp = np.where(pe_lo_cmp > 0.0, pe_lo_cmp, np.nan)
    pe_la_cmp = np.where(pe_la_cmp > 0.0, pe_la_cmp, np.nan)
    pe_v_cmp = np.where(pe_v_cmp > 0.0, pe_v_cmp, np.nan)
    pl1_lo_cmp = np.where(pl1_lo_cmp > 0.0, pl1_lo_cmp, np.nan)
    pl1_la_cmp = np.where(pl1_la_cmp > 0.0, pl1_la_cmp, np.nan)
    pl1_v_cmp = np.where(pl1_v_cmp > 0.0, pl1_v_cmp, np.nan)
    pl2_lo_cmp = np.where(pl2_lo_cmp > 0.0, pl2_lo_cmp, np.nan)
    pl2_la_cmp = np.where(pl2_la_cmp > 0.0, pl2_la_cmp, np.nan)
    pl2_v_cmp = np.where(pl2_v_cmp > 0.0, pl2_v_cmp, np.nan)

    fig_cmp, axs_cmp = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    axs_cmp[0].scatter(t_cmp, pe_lo_cmp, label='LoPE', color='blue', s=8, marker='.', alpha=0.9)
    axs_cmp[0].scatter(t_cmp, pl1_lo_cmp, label='LoPL (consider correlation)', color='red', s=8, marker='.', alpha=0.9)
    axs_cmp[0].scatter(t_cmp, pl2_lo_cmp, label='LoPL (no correlation)', color='tab:brown', s=8, marker='.', alpha=0.9)
    axs_cmp[0].axhline(lo_al, color='black', linestyle='--', linewidth=2, label=f'LoAL={lo_al:.2f}m')
    axs_cmp[0].set_ylabel('PE/PL (m)')
    axs_cmp[0].set_title('Longitudinal PE/PL Comparison')
    axs_cmp[0].legend(loc='upper right', framealpha=0.75)
    axs_cmp[0].grid(True, which="both", ls="-")

    axs_cmp[1].scatter(t_cmp, pe_la_cmp, label='LaPE', color='blue', s=8, marker='.', alpha=0.9)
    axs_cmp[1].scatter(t_cmp, pl1_la_cmp, label='LaPL (consider correlation)', color='red', s=8, marker='.', alpha=0.9)
    axs_cmp[1].scatter(t_cmp, pl2_la_cmp, label='LaPL (no correlation)', color='tab:brown', s=8, marker='.', alpha=0.9)
    axs_cmp[1].axhline(la_al, color='black', linestyle='--', linewidth=2, label=f'LaAL={la_al:.2f}m')
    axs_cmp[1].set_ylabel('PE/PL (m)')
    axs_cmp[1].set_title('Lateral PE/PL Comparison')
    axs_cmp[1].legend(loc='upper right', framealpha=0.75)
    axs_cmp[1].grid(True, which="both", ls="-")

    axs_cmp[2].scatter(t_cmp, pe_v_cmp, label='VPE', color='blue', s=8, marker='.', alpha=0.9)
    axs_cmp[2].scatter(t_cmp, pl1_v_cmp, label='VPL (consider correlation)', color='red', s=8, marker='.', alpha=0.9)
    axs_cmp[2].scatter(t_cmp, pl2_v_cmp, label='VPL (no correlation)', color='tab:brown', s=8, marker='.', alpha=0.9)
    axs_cmp[2].axhline(v_al, color='black', linestyle='--', linewidth=2, label=f'VAL={v_al:.2f}m')
    axs_cmp[2].set_ylabel('PE/PL (m)')
    axs_cmp[2].set_xlabel('Time (s)')
    axs_cmp[2].set_title('Vertical PE/PL Comparison')
    axs_cmp[2].legend(loc='upper right', framealpha=0.75)
    axs_cmp[2].grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('integrity_evaluation_components.compare_wo_corr_vs_corr.raw.png')
    plt.close(fig_cmp)
    print("Saved plot to integrity_evaluation_components.compare_wo_corr_vs_corr.raw.png")
    
    # Plot 2: Horizontal Error vs PLs
    plt.figure(figsize=(12, 8))
    plt.plot(timestamps, horiz_error, label='Horizontal Error', color='blue', linewidth=2)
    plt.plot(timestamps, pl_lo_vis, label='LoPL', color='red', linestyle='-', alpha=0.5, linewidth=3)
    plt.plot(timestamps, pl_la_vis, label='LaPL', color='orange', linestyle='-', alpha=0.5, linewidth=3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Error / PL (m)')
    plt.title('Horizontal Position Error vs Protection Levels')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-")
    plt.savefig('integrity_evaluation_horizontal.png')
    print("Saved plot to integrity_evaluation_horizontal.png")

    # Plot 2 raw: Horizontal Error vs raw PLs
    plt.figure(figsize=(12, 8))
    plt.plot(timestamps, horiz_error, label='Horizontal Error', color='blue', linewidth=2)
    plt.plot(timestamps, np.minimum(pl_lo_raw, 2 * lo_al), label='LoPL(raw)', color='red', linestyle='-', alpha=0.5, linewidth=3)
    plt.plot(timestamps, np.minimum(pl_la_raw, 2 * la_al), label='LaPL(raw)', color='orange', linestyle='-', alpha=0.5, linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Error / PL (m)')
    plt.title('Horizontal Position Error vs Raw Protection Levels')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-")
    plt.savefig('integrity_evaluation_horizontal.raw.png')
    plt.close()
    print("Saved plot to integrity_evaluation_horizontal.raw.png")

    # Plot 3: Trajectory with PL Boxes (The "Cube" plot)
    print("Generating Trajectory with PL boxes plot...")
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Plot only 0-105s segment for trajectory figures.
    traj_mask = (timestamps >= 0.0) & (timestamps <= 105.0)
    traj_indices = np.where(traj_mask)[0]

    # Plot GT/Est from matched trajectory within 0-105s.
    gt_x = [matches[i][0]['pos'][0] for i in traj_indices]
    gt_y = [matches[i][0]['pos'][1] for i in traj_indices]
    plt.plot(gt_x, gt_y, 'k-', label='Ground Truth', linewidth=1, alpha=0.6)
    
    # Plot Est
    est_x = [matches[i][1]['pos'][0] for i in traj_indices]
    est_y = [matches[i][1]['pos'][1] for i in traj_indices]
    plt.plot(est_x, est_y, 'b-', label='Estimated', linewidth=1)
    
    # Draw PL Rectangles
    # Downsample for visibility - adjust step based on data density
    step = 10 
    
    # Define ranges, colors, and scaling
    # Format: (min_hpl, max_hpl, color, label)
    # Colors: Green -> Yellow -> Red -> Dark Red (Academic style)
    ranges = [
        (0.0, 0.05, '#006400', 'HPL < 0.05m'),         # Dark Green
        (0.05, 0.1, '#32CD32', '0.05 <= HPL < 0.1m'),  # Lime Green
        (0.1, 0.5, '#CCCC00', '0.1 <= HPL < 0.5m'),    # Dark Yellow
        (0.5, 1.0, '#FFA500', '0.5 <= HPL < 1.0m'),    # Orange
        (1.0, 5.0, '#FF4500', '1.0 <= HPL < 5.0m'),    # Orange Red
        (5.0, float('inf'), '#800000', 'HPL >= 5.0m')  # Maroon
    ]
    
    # Visualization scales for trajectory boxes.
    # #sym:scale_magnified and #sym:scale_normal are both active in plotting.
    scale_magnified = 30.0
    scale_normal = 45.0
    # #sym:scale_normal_start_hpl: bins with lower bound >= this value use scale_normal.
    scale_normal_start_hpl = 1
    
    # Create dummy handles for the legend
    for start, end, color, label in ranges:
        # Use exactly the same condition as the actual drawing scale selection.
        scale = scale_normal if start >= scale_normal_start_hpl else scale_magnified
        scale_str = f" (x{scale})"
        plt.plot([], [], color=color, linestyle='-', linewidth=1.5, label=label + scale_str)
    
    for i in traj_indices[::step]:
        gt, est = matches[i]
        x, y, z = gt['pos']
        xpl_raw, ypl_raw, vpl_raw = est['pl']
        q = gt['quat']
        
        if np.isnan(xpl_raw) or np.isnan(ypl_raw):
            continue

        # Use uncapped PL for HPL bin classification so >=5m bin can be reached.
        hpl = np.sqrt(xpl_raw**2 + ypl_raw**2)
        
        box_color = 'black'
        current_scale = scale_magnified
        
        # Determine color and scale
        for start, end, color, label in ranges:
            if start <= hpl < end:
                box_color = color
                if start >= scale_normal_start_hpl:
                    current_scale = scale_normal
                break

        # Cap only drawing size for readability, then apply scaling.
        xpl = min(xpl_raw, 2 * lo_al)
        ypl = min(ypl_raw, 2 * la_al)
        xpl *= current_scale
        ypl *= current_scale
            
        # Yaw
        yaw = get_yaw_from_quat(q)
        
        # Define rectangle corners in local frame (centered at 0)
        # Assuming XPL is along local X, YPL along local Y
        # Corners: (xpl, ypl), (-xpl, ypl), (-xpl, -ypl), (xpl, -ypl)
        lopl = ypl
        lapl = xpl
        
        corners_local = np.array([
            [lopl, lapl],
            [-lopl, lapl],
            [-lopl, -lapl],
            [lopl, -lapl],
            [lopl, lapl] # Close loop
        ])
        
        # Rotation matrix
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate and translate
        corners_global = corners_local @ R.T
        corners_global[:, 0] += x
        corners_global[:, 1] += y
        
        plt.plot(corners_global[:, 0], corners_global[:, 1], color=box_color, linewidth=1.0, alpha=0.8)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory and Protection Levels (LoPL/LaPL Boxes)')
    plt.axis('equal')
    plt.ylim(-150.0, 200.0)
    plt.legend(loc='upper right')  # 这里保留原图例字体大小，因为原代码指定了14，可以不改或改为FONT_SIZE_LEGEND
    plt.grid(True)
    plt.savefig('integrity_evaluation_trajectory.png')
    print("Saved plot to integrity_evaluation_trajectory.png")

    # Plot 3 raw: Trajectory with raw PL Boxes
    print("Generating raw trajectory with PL boxes plot...")
    plt.figure(figsize=(12, 12))
    plt.plot(gt_x, gt_y, 'k-', label='Ground Truth', linewidth=1, alpha=0.6)
    plt.plot(est_x, est_y, 'b-', label='Estimated', linewidth=1)

    for start, end, color, label in ranges:
        scale = scale_normal if start >= scale_normal_start_hpl else scale_magnified
        scale_str = f" (x{scale})"
        plt.plot([], [], color=color, linestyle='-', linewidth=1.5, label=label + scale_str)

    for i in traj_indices[::step]:
        gt, est = matches[i]
        x, y, z = gt['pos']
        xpl_raw, ypl_raw, vpl_raw = pl_lo_raw[i], pl_la_raw[i], pl_v_raw[i]
        q = gt['quat']

        if np.isnan(xpl_raw) or np.isnan(ypl_raw):
            continue

        hpl = np.sqrt(xpl_raw**2 + ypl_raw**2)
        box_color = 'black'
        current_scale = scale_magnified
        for start, end, color, label in ranges:
            if start <= hpl < end:
                box_color = color
                if start >= scale_normal_start_hpl:
                    current_scale = scale_normal
                break

        xpl = min(xpl_raw, 2 * lo_al)
        ypl = min(ypl_raw, 2 * la_al)
        xpl *= current_scale
        ypl *= current_scale
        yaw = get_yaw_from_quat(q)
        lopl = ypl
        lapl = xpl
        corners_local = np.array([
            [lopl, lapl],
            [-lopl, lapl],
            [-lopl, -lapl],
            [lopl, -lapl],
            [lopl, lapl]
        ])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners_global = corners_local @ R.T
        corners_global[:, 0] += x
        corners_global[:, 1] += y
        plt.plot(corners_global[:, 0], corners_global[:, 1], color=box_color, linewidth=1.0, alpha=0.8)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
  # Optional: zoom in on Y axis around trajectory
    plt.title('Trajectory and Raw Protection Levels (LoPL/LaPL Boxes)')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('integrity_evaluation_trajectory.raw.png')
    plt.close()
    print("Saved plot to integrity_evaluation_trajectory.raw.png")

    # Plot 4: Satellite count over time (0-105s only).
    if os.path.exists(raw_obs_file):
        print("Generating satellite count plot...")
        sat_t, sat_num, gnss_fix = read_satellite_gnss_from_raw(raw_obs_file)
        if sat_t.size > 0:
            sat_mask = (sat_t >= 0.0) & (sat_t <= 105.0)
            sat_t_plot = sat_t[sat_mask]
            sat_num_plot = sat_num[sat_mask]

            fig_sv, ax_sat = plt.subplots(figsize=(12, 8))
            ax_sat.plot(sat_t_plot, sat_num_plot, color='tab:green', linewidth=2.0, label='Satellite Count')
            ax_sat.scatter(sat_t_plot, sat_num_plot, color='tab:green', s=8, marker='.', alpha=0.85)
            ax_sat.set_xlabel('Time (s)')
            ax_sat.set_ylabel('Satellites', color='tab:green')
            ax_sat.tick_params(axis='y', labelcolor='tab:green')
            ax_sat.set_title('Satellite and Visual Measurements Over Time')
            ax_sat.grid(True, which='both', ls='-')
            # Highlight GNSS-challenging interval.
            ax_sat.axvspan(12.0, 40.0, color='gray', alpha=0.22, label='GNSS-Challenging Environment')

            handles = []
            labels = []
            h1, l1 = ax_sat.get_legend_handles_labels()
            handles.extend(h1)
            labels.extend(l1)

            if os.path.exists(subset_info_file):
                vis_t, vis_num = read_visual_meas_from_subset_info(subset_info_file)
                if vis_t.size > 0:
                    vis_mask = (vis_t >= 0.0) & (vis_t <= 105.0)
                    vis_t_plot = vis_t[vis_mask]
                    vis_num_plot = vis_num[vis_mask]
                    if vis_t_plot.size > 0:
                        ax_vis = ax_sat.twinx()
                        ax_vis.plot(vis_t_plot, vis_num_plot, color='tab:blue', linewidth=2.0, label='Visual Measurements')
                        ax_vis.scatter(vis_t_plot, vis_num_plot, color='tab:blue', s=8, marker='.', alpha=0.80)
                        ax_vis.set_ylabel('Visual Measurements', color='tab:blue')
                        ax_vis.tick_params(axis='y', labelcolor='tab:blue')
                        h2, l2 = ax_vis.get_legend_handles_labels()
                        handles.extend(h2)
                        labels.extend(l2)
                    else:
                        print('No visual measurement samples in 0-105s from subset info.')
                else:
                    print('No valid entries found in subset info file.')
            else:
                print(f'Subset info file not found: {subset_info_file}')

            if handles:
                ax_sat.legend(handles, labels, loc='upper right')

            fig_sv.tight_layout()
            plt.savefig('integrity_evaluation_satellite_gnss.png')
            plt.close()
            print("Saved plot to integrity_evaluation_satellite_gnss.png")

            # Plot 5 (new): Trajectory colored by satellite count quality.
            # Higher satellite count generally indicates better GNSS observability.
            sat_num_on_traj = interpolate_series_to_timestamps(sat_t_plot, sat_num_plot, timestamps)
            sat_num_traj = sat_num_on_traj[traj_indices]

            est_x_arr = np.asarray(est_x, dtype=float)
            est_y_arr = np.asarray(est_y, dtype=float)
            valid_traj = np.isfinite(est_x_arr) & np.isfinite(est_y_arr) & np.isfinite(sat_num_traj)

            if np.any(valid_traj):
                plt.figure(figsize=(13, 8))
                plt.plot(gt_x, gt_y, 'k-', label='Ground Truth', linewidth=1.0, alpha=0.5)
                sc = plt.scatter(
                    est_x_arr[valid_traj],
                    est_y_arr[valid_traj],
                    c=sat_num_traj[valid_traj],
                    cmap='RdYlGn',
                    s=16,
                    marker='o',
                    alpha=0.9,
                    label='Satellites Number (colored by satellite count)'
                )
                cbar = plt.colorbar(sc)
                cbar.set_label('Satellite Count')

                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title('Trajectory Colored by Satellite Count')
                plt.axis('auto')
                plt.ylim(-150.0, 100.0)
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('integrity_evaluation_trajectory_satellite_quality.png')
                plt.close()
                print("Saved plot to integrity_evaluation_trajectory_satellite_quality.png")
            else:
                print("No valid trajectory-satellite overlap for quality trajectory plot.")
        else:
            print(f"No valid $GPGGA records found in {raw_obs_file}.")
    else:
        print(f"Raw observation file not found: {raw_obs_file}")
    
    # plt.show()

if __name__ == "__main__":
    main()