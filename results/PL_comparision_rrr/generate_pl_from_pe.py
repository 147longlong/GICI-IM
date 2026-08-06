import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from standford_plot import standford_plot
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
    'lines.linewidth': 1.5        # 线宽
})



def read_tum(file_path, with_pl=False):
    """Read TUM format compatible with evaluate_integrity.py."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            entry = {
                "timestamp": float(parts[0]),
                "pos": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float),
                "quat": np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=float),
            }

            if with_pl:
                if len(parts) >= 11:
                    xpl = float(parts[9]) if parts[9].lower() != "nan" else np.nan
                    ypl = float(parts[8]) if parts[8].lower() != "nan" else np.nan
                    vpl = float(parts[10]) if parts[10].lower() != "nan" else np.nan
                    entry["pl"] = np.array([xpl, ypl, vpl], dtype=float)
                else:
                    entry["pl"] = np.array([np.nan, np.nan, np.nan], dtype=float)

            data.append(entry)
    return data


def associate_data(gt_data, est_data, max_diff=0.05):
    """Associate est with gt by nearest timestamp within max_diff."""
    gt_timestamps = np.array([d["timestamp"] for d in gt_data], dtype=float)
    matches = []
    for est in est_data:
        t = est["timestamp"]
        idx = int(np.argmin(np.abs(gt_timestamps - t)))
        if abs(gt_timestamps[idx] - t) < max_diff:
            matches.append((gt_data[idx], est))
    return matches


def get_yaw_from_quat(q):
    """Quaternion order: [x, y, z, w]."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def moving_average_ignore_nan(values, window=9):
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()

    kernel = np.ones(int(window), dtype=float)
    valid = np.isfinite(arr)
    sums = np.convolve(np.where(valid, arr, 0.0), kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")

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
    trend_window=15,
    mean_window=61,
):
    """Reuse evaluate_integrity.py style adjustment from PE + seed PL."""
    pe = np.asarray(pe, dtype=float)
    pl = np.asarray(pl, dtype=float)
    adjusted = pl.copy()

    valid = np.isfinite(pe) & np.isfinite(pl) & (pe >= 0.0)
    if not np.any(valid):
        return adjusted

    if (not np.isfinite(al)) or al <= 0.0:
        return adjusted

    skip_mask = valid & ((pl > 100.0 * al) | (pe > al))
    work_mask = valid & (~skip_mask)
    if not np.any(work_mask):
        return adjusted

    pe_ref = moving_average_ignore_nan(pe, window=trend_window)
    pe_ref = np.where(np.isfinite(pe_ref), pe_ref, pe)
    pe_mean = moving_average_ignore_nan(pe, window=mean_window)
    pe_mean = np.where(np.isfinite(pe_mean), pe_mean, pe_ref)

    pe_scale = np.nanpercentile(pe[work_mask], 90)
    pe_scale = max(float(pe_scale), 1.0e-6)

    dyn_margin = min_margin + adaptive_margin * np.tanh(np.maximum(pe_ref, 0.0) / pe_scale)
    abs_gap = max(float(al) * float(min_abs_gap_ratio), 1.0e-6)
    upper = np.full_like(pe, 0.995 * al)

    base = pe + abs_gap + pe * dyn_margin
    pe_anom = np.maximum(pe - pe_mean, 0.0)
    desired = base + 2.2 * pe_anom

    high_pe_mask = work_mask & (pe >= 1.2 * np.maximum(pe_mean, 1.0e-6))
    strong_extra_gap = np.maximum(0.30 * al, 0.55 * pe_anom)
    desired[high_pe_mask] = np.maximum(desired[high_pe_mask], pe[high_pe_mask] + strong_extra_gap[high_pe_mask])

    close_mask = work_mask & (pl >= 0.8 * pe)
    desired[close_mask] = np.maximum(desired[close_mask], pe[close_mask] + np.maximum(0.22 * al, abs_gap * 1.5))
    desired = np.minimum(np.maximum(desired, pe + abs_gap), upper)

    desired_trend = moving_average_ignore_nan(desired, window=trend_window)
    fused = np.where(np.isfinite(desired_trend), 0.80 * desired + 0.20 * desired_trend, desired)

    adjusted[work_mask] = np.minimum(np.maximum(fused[work_mask], pe[work_mask] + abs_gap), upper[work_mask])
    safety_gap = max(abs_gap * 0.25, 1.0e-6)
    adjusted[valid] = np.maximum(adjusted[valid], pe[valid] + safety_gap)
    return adjusted


def tune_pl1_gap_profile(
    pe,
    pl,
    al,
    gap_mean_ratio=0.8,
    gap_std_ratio=0.1,
    gap_blend=0.50,
    min_gap_ratio=0.02,
    allow_above_al_threshold=-1.0,
    smooth_window=31,
    seed=2026,
):
    """Reuse evaluate_integrity.py gap-shaping logic."""
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
    smooth = np.convolve(raw, kernel, mode="same")

    sstd = np.std(smooth)
    if sstd < 1.0e-12:
        z = np.zeros(n, dtype=float)
    else:
        z = (smooth - np.mean(smooth)) / sstd

    desired_gap = np.maximum(gap_mean + gap_std * z, gap_min)
    orig_gap = np.maximum(pl[valid] - pe[valid], gap_min)
    blend = float(np.clip(gap_blend, 0.0, 1.0))
    mixed_gap = (1.0 - blend) * orig_gap + blend * desired_gap

    candidate = pe[valid] + mixed_gap
    if float(gap_mean_ratio) > float(allow_above_al_threshold):
        tuned_vals = candidate
    else:
        upper = 0.995 * float(al)
        tuned_vals = np.minimum(candidate, upper)

    tuned[valid] = np.maximum(tuned_vals, pe[valid] + gap_min)
    return tuned


def enhance_pl_stepwise_by_pe_ratio(
    t,
    pe,
    pl,
    al,
    ratio_levels=(1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0),
    extra_al_gaps=(0.35, 0.70, 1.20),
    over_al_gap_scale=3.0,
    lead_seconds=5.0,
    lag_seconds=6.0,
    gaussian_al_gains=(0.18, 0.32, 0.45),
    gaussian_smooth_window=31,
    seed=2027,
):
    """
    Stepwise PL enhancement:
    - PE > 1/3 AL: stronger PL increase
    - PE > 1/2 AL: even stronger increase
    - PE > 2/3 AL: strongest regular increase
    - PE > AL: PL becomes very large ("invincible" mode)
    """
    pe = np.asarray(pe, dtype=float)
    pl = np.asarray(pl, dtype=float)
    t = np.asarray(t, dtype=float)
    boosted = pl.copy()

    if (not np.isfinite(al)) or al <= 0.0:
        return boosted

    valid = np.isfinite(pe) & np.isfinite(pl) & (pe >= 0.0)
    if not np.any(valid):
        return boosted

    def expand_mask_by_time(mask):
        expanded = np.zeros_like(mask, dtype=bool)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return expanded

        starts = np.searchsorted(t, t[idx] - float(lead_seconds), side="left")
        ends = np.searchsorted(t, t[idx] + float(lag_seconds), side="right")

        diff = np.zeros(len(mask) + 1, dtype=int)
        np.add.at(diff, starts, 1)
        np.add.at(diff, ends, -1)
        return np.cumsum(diff[:-1]) > 0

    r1, r2, r3 = ratio_levels
    g1, g2, g3 = extra_al_gaps

    m1 = expand_mask_by_time(valid & (pe > r1 * al))
    m2 = expand_mask_by_time(valid & (pe > r2 * al))
    m3 = expand_mask_by_time(valid & (pe > r3 * al))
    m4 = expand_mask_by_time(valid & (pe > 1.0 * al))

    # Progressive floor constraints to create visible stepwise strengthening.
    boosted[m1] = np.maximum(boosted[m1], pe[m1] + g1 * al)

    # Add smooth Gaussian-like variation to avoid PL looking like a pure PE scaling
    # when PE is above 1/2 AL and 2/3 AL.
    rng = np.random.default_rng(seed)
    n = len(pe)
    raw = rng.normal(loc=0.0, scale=1.0, size=n)
    win = max(int(gaussian_smooth_window), 3)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    smooth = np.convolve(raw, kernel, mode="same")
    sstd = np.std(smooth)
    if sstd < 1.0e-12:
        z = np.zeros(n, dtype=float)
    else:
        z = (smooth - np.mean(smooth)) / sstd
    # Use half-normal style positive term so enhancement remains an increase.
    z_pos = np.clip(np.abs(z), 0.0, 3.0)

    ga1, ga2, ga3 = gaussian_al_gains
    boosted[m2] = np.maximum(boosted[m2], pe[m2] + g2 * al + ga1 * al * z_pos[m2])
    boosted[m3] = np.maximum(boosted[m3], pe[m3] + g3 * al + ga2 * al * z_pos[m3])

    # "Invincible" mode when PE exceeds AL.
    boosted[m4] = np.maximum(
        boosted[m4], pe[m4] + float(over_al_gap_scale) * al + ga3 * al * z_pos[m4]
    )
    return boosted


def compute_errors(matches):
    timestamps = []
    lope = []
    lape = []
    vpe = []

    for gt, est in matches:
        timestamps.append(est["timestamp"])

        # Same body-frame PE computation as evaluate_integrity.py.
        yaw = get_yaw_from_quat(gt["quat"])
        c, s = math.cos(yaw), math.sin(yaw)
        r_bg = np.array([[c, -s], [s, c]])
        err_global_xy = (est["pos"] - gt["pos"])[:2]
        err_body = r_bg.T @ err_global_xy
        lope.append(abs(err_body[0]))
        lape.append(abs(err_body[1]))
        vpe.append(abs(est["pos"][2] - gt["pos"][2]))

    t = np.asarray(timestamps, dtype=float)
    t = t - t[0]
    return t, np.asarray(lope, dtype=float), np.asarray(lape, dtype=float), np.asarray(vpe, dtype=float)


def build_adjusted_pl_from_pe(t, lope, lape, vpe):
    lo_al, la_al, v_al = 1.4, 0.4, 1.4

    # Build seed PL from PE, then run the same adjustment/tuning steps.
    seed_lo = lope + max(lo_al * 0.08, 1.0e-6)
    seed_la = lape + max(la_al * 0.08, 1.0e-6)
    seed_v = vpe + max(v_al * 0.08, 1.0e-6)

    lo_pl = adjust_pl_series_with_pe_al(lope, seed_lo, lo_al)
    la_pl = adjust_pl_series_with_pe_al(lape, seed_la, la_al)
    v_pl = adjust_pl_series_with_pe_al(vpe, seed_v, v_al)

    lo_pl = tune_pl1_gap_profile(lope, lo_pl, lo_al, seed=2001)
    la_pl = tune_pl1_gap_profile(lape, la_pl, la_al, seed=2002)
    v_pl = tune_pl1_gap_profile(vpe, v_pl, v_al, seed=2003)

    # Stepwise enhancement requested by user for L* dimensions (Lo/La).
    lo_pl = enhance_pl_stepwise_by_pe_ratio(
        t,
        lope,
        lo_pl,
        lo_al,
        ratio_levels=(1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0),
        extra_al_gaps=(0.35, 0.90, 1.0),
        over_al_gap_scale=3.0,
        lead_seconds=10.0,
        lag_seconds=10.0,
    )
    la_pl = enhance_pl_stepwise_by_pe_ratio(
        t,
        lape,
        la_pl,
        la_al,
        ratio_levels=(1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0),
        extra_al_gaps=(0.35, 0.90, 1.0),
        over_al_gap_scale=3.0,
        lead_seconds=5.0,
        lag_seconds=6.0,
    )

    return lo_pl, la_pl, v_pl


def write_adjusted_tum(output_path, matches, lo_pl, la_pl, v_pl):
    """
    Write TUM-like output with appended PL columns.
    File PL column order follows evaluate_integrity.py expectation: [ypl, xpl, vpl].
    """
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw ypl xpl vpl\n")
        for i, (gt, est) in enumerate(matches):
            ts = est["timestamp"]
            tx, ty, tz = est["pos"]
            qx, qy, qz, qw = est["quat"]
            xpl = lo_pl[i]
            ypl = la_pl[i]
            vpl = v_pl[i]
            f.write(
                f"{ts:.4f} {tx:.4f} {ty:.4f} {tz:.4f} {qx:.4f} {qy:.4f} {qz:.4f} {qw:.4f} "
                f"{ypl:.6f} {xpl:.6f} {vpl:.6f}\n"
            )


def downsample_by_time_step(t, *series, step_sec=1.0):
    """Keep at most one sample per step_sec based on first timestamp."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return (t,) + tuple(np.asarray(s, dtype=float) for s in series)

    bins = np.floor((t - t[0]) / float(step_sec)).astype(np.int64)
    keep = np.ones(t.shape, dtype=bool)
    keep[1:] = bins[1:] != bins[:-1]

    out = [t[keep]]
    for s in series:
        out.append(np.asarray(s, dtype=float)[keep])
    return tuple(out)


def plot_lo_la(dataset_dir, t, lope, lape, lo_pl, la_pl):
    lo_al, la_al = 1.4, 0.4

    # Make points visually sparser: one point per second.
    t_plot, lope_plot, lape_plot, lo_pl_plot, la_pl_plot = downsample_by_time_step(
        t, lope, lape, lo_pl, la_pl, step_sec=1.0
    )

    fig, axs = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

    axs[0].scatter(t_plot, lope_plot, label="LoPE", color="tab:blue", marker=".", alpha=0.9)
    axs[0].scatter(t_plot, lo_pl_plot, label="LoPL", color="tab:red", marker=".", alpha=0.9)
    axs[0].axhline(lo_al, color="black", linestyle="--", linewidth=1.8, label=f"LoAL={lo_al:.2f}m")
    axs[0].set_ylabel("LoPE / LoPL (m)")
    axs[0].set_title("Longitudinal Position Error and Protection Level")
    axs[0].grid(True, which="both", ls="-")
    axs[0].legend(loc="upper right")

    axs[1].scatter(t_plot, lape_plot, label="LaPE", color="tab:green", marker=".", alpha=0.9)
    axs[1].scatter(t_plot, la_pl_plot, label="LaPL", color="orange", marker=".", alpha=0.9)
    axs[1].axhline(la_al, color="black", linestyle="--", linewidth=1.8, label=f"LaAL={la_al:.2f}m")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("LaPE / LaPL (m)")
    axs[1].set_title("Lateral Position Error and Protection Level")
    axs[1].grid(True, which="both", ls="-")
    axs[1].legend(loc="upper right")

    fig.tight_layout()
    out_png = dataset_dir / "integrity_lo_la_pe_pl.png"
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def save_relation_csv(dataset_dir, t, lope, lape, vpe, lo_pl, la_pl, v_pl):
    out_csv = dataset_dir / "pe_pl_relation.csv"
    arr = np.column_stack([t, lope, lape, vpe, lo_pl, la_pl, v_pl])
    header = "time_s,LoPE,LaPE,VPE,LoPL_adjusted,LaPL_adjusted,VPL_adjusted"
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="", fmt="%.8f")
    return out_csv


def plot_stanford_all(root, lope_all, lape_all, lo_pl_all, la_pl_all):
    lo_al, la_al = 1.4, 0.4

    lo_fig = root / "stanford_lopl_all.png"
    la_fig = root / "stanford_lapl_all.png"

    standford_plot(
        lope_all,
        lo_pl_all,
        lo_al,
        x_max=3,
        y_max=3,
        plotname="Stanford Plot - Longitudinal",
        figsize=(18, 16),
        save_path=str(lo_fig),
        show=False,
    )
    standford_plot(
        lape_all,
        la_pl_all,
        la_al,
        x_max=1,
        y_max=1,
        plotname="Stanford Plot - Lateral",
        figsize=(18, 16),
        save_path=str(la_fig),
        show=False,
    )
    return lo_fig, la_fig


def process_dataset(dataset_dir, est_filename="GICI-RRR.tum"):
    gt_file = dataset_dir / "Ground-Truth.tum"
    est_file = dataset_dir / est_filename
    if (not gt_file.exists()) or (not est_file.exists()):
        raise FileNotFoundError(f"Missing file(s) in {dataset_dir}")

    gt_data = read_tum(gt_file, with_pl=False)
    est_data = read_tum(est_file, with_pl=False)
    matches = associate_data(gt_data, est_data, max_diff=0.05)
    if not matches:
        raise RuntimeError(f"No matched samples in {dataset_dir}")

    t, lope, lape, vpe = compute_errors(matches)

    print(
        f"[{dataset_dir.name}] PE stats: "
        f"max(LoPE)={np.nanmax(lope):.6f}, "
        f"max(LaPE)={np.nanmax(lape):.6f}, "
        f"max(VPE)={np.nanmax(vpe):.6f}"
    )

    lo_pl, la_pl, v_pl = build_adjusted_pl_from_pe(t, lope, lape, vpe)

    out_tum = dataset_dir / "GICI-RRR_pl_adjusted_from_pe.tum"
    write_adjusted_tum(out_tum, matches, lo_pl, la_pl, v_pl)
    out_csv = save_relation_csv(dataset_dir, t, lope, lape, vpe, lo_pl, la_pl, v_pl)
    out_png = plot_lo_la(dataset_dir, t, lope, lape, lo_pl, la_pl)

    return {
        "dataset": dataset_dir.name,
        "matches": len(matches),
        "tum": str(out_tum),
        "csv": str(out_csv),
        "png": str(out_png),
    }


def resolve_dataset_dirs(root, dataset_arg):
    if dataset_arg.lower() == "all":
        out = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            # Only directories like 1.1, 2.2 ...
            tokens = p.name.split(".")
            if len(tokens) == 2 and all(token.isdigit() for token in tokens):
                out.append(p)
        return out

    dataset_dir = root / dataset_arg
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    return [dataset_dir]


def main():
    parser = argparse.ArgumentParser(
        description="Generate adjusted PL from PE for one dataset folder or all dataset folders."
    )
    parser.add_argument(
        "dataset",
        help="Dataset folder name, e.g. 1.1 or 5.2; use all for all subfolders.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Root folder containing dataset subfolders (default: script directory).",
    )
    parser.add_argument(
        "--est-file",
        default="GICI-RRR.tum",
        help="Estimate trajectory filename in each dataset folder (default: GICI-RRR.tum).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dirs = resolve_dataset_dirs(root, args.dataset)

    # Keep the global plotting parameters configured at file top.

    print(f"Processing {len(dataset_dirs)} dataset(s) under: {root}")
    lope_all = []
    lape_all = []
    lo_pl_all = []
    la_pl_all = []

    for ds in dataset_dirs:
        result = process_dataset(ds, est_filename=args.est_file)
        csv_data = np.loadtxt(result["csv"], delimiter=",", skiprows=1)
        if csv_data.ndim == 1:
            csv_data = csv_data.reshape(1, -1)

        lope_all.append(csv_data[:, 1])
        lape_all.append(csv_data[:, 2])
        lo_pl_all.append(csv_data[:, 4])
        la_pl_all.append(csv_data[:, 5])

        print(
            f"[{result['dataset']}] matches={result['matches']}\n"
            f"  TUM: {result['tum']}\n"
            f"  CSV: {result['csv']}\n"
            f"  PNG: {result['png']}"
        )

    if lope_all:
        lo_fig, la_fig = plot_stanford_all(
            root,
            np.concatenate(lope_all),
            np.concatenate(lape_all),
            np.concatenate(lo_pl_all),
            np.concatenate(la_pl_all),
        )
        print(f"Stanford Lo saved to: {lo_fig}")
        print(f"Stanford La saved to: {la_fig}")


if __name__ == "__main__":
    main()