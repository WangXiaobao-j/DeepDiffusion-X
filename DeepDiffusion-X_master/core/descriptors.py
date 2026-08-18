"""Extraction of the nine core pathway descriptors from the MEP and barrier files."""
import os
import re
import warnings
import numpy as np
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

kBT = 2.478  # kJ/mol @ 298 K

BOUNDARY_MAD_FACTOR = 5.0
BOUNDARY_N_SMOOTH = 3

# The 9 core descriptors retained in the final output (path-geometry
# related keys, matching compute_descriptors()'s output dict exactly)
CORE_DESCRIPTOR_KEYS = [
    "barrier_official", "sigma_E", "eta_main", "tortuosity", "E_max",
    "E_min_official", "peak_slope_mean", "H_entropy", "barrier_freq",
]
# Derived from path-file geometry (excludes barrier_official / E_min_official, which come from the barrier file)
_PATH_DERIVED_KEYS = ["sigma_E", "eta_main", "tortuosity", "E_max",
                       "peak_slope_mean", "H_entropy", "barrier_freq"]


# ====================================================================
# 1. File reading (matches the reference implementation)
# ====================================================================
def read_path_file(fpath: str):
    meta, rows = {}, []
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    m = re.search(r"n_nodes\s*=\s*(\d+)", line)
                    if m: meta["n_nodes"] = int(m.group(1))
                    m = re.search(r"barrier\s*=\s*([-\d.]+)", line)
                    if m: meta["barrier"] = float(m.group(1))
                    m = re.search(r"n_supercell\s*=\s*(\d+)", line)
                    if m: meta["n_supercell"] = int(m.group(1))
                else:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            rows.append([float(p) for p in parts[:4]])
                        except ValueError:
                            pass
    except Exception:
        return None, None, {}

    if len(rows) < 5:
        return None, None, meta

    arr = np.array(rows)
    return arr[:, :3], arr[:, 3], meta


def read_barriers_file(folder: str, name: str) -> dict:
    fpath = os.path.join(folder, f"{name}_diffusion_barriers.txt")
    result = {ax: {"traverses": False} for ax in ("X", "Y", "Z")}
    if not os.path.exists(fpath):
        return result
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        for ax in ("X", "Y", "Z"):
            pat = (rf"{ax} direction:\s*barrier\s*=\s*([-\d.]+)\s*kJ/mol"
                   rf".*?transition energy\s*=\s*([-\d.]+)\s*kJ/mol"
                   rf"(?:.*?path nodes\s*=\s*(\d+))?")
            m = re.search(pat, txt)
            if m:
                result[ax] = {
                    "traverses": True,
                    "barrier": float(m.group(1)),
                    "transition_energy": float(m.group(2)),
                    "n_nodes_barrier": int(m.group(3)) if m.group(3) else None,
                }
    except Exception:
        pass
    return result


def read_summary(folder: str, name: str) -> dict:
    fpath = os.path.join(folder, f"{name}_analysis_summary.txt")
    info = {}
    if not os.path.exists(fpath):
        return info
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        m = re.search(r"Dimensionality:\s*(\dD)", txt)
        if m:
            info["Dimensionality"] = m.group(1)
        for ax in ("X", "Y", "Z"):
            m = re.search(rf"{ax}\s*=\s*(Yes|No)", txt, re.IGNORECASE)
            if m:
                info[f"traverses_{ax}"] = (m.group(1).lower() == "yes")
        m = re.search(r"Original unit cell:.*?a=([\d.]+).*?b=([\d.]+).*?c=([\d.]+)", txt)
        if m:
            info["a_A"] = float(m.group(1))
            info["b_A"] = float(m.group(2))
            info["c_A"] = float(m.group(3))
    except Exception:
        pass
    return info


# ====================================================================
# 2. Cell-boundary artifact smoothing (matches the reference implementation)
# ====================================================================
def smooth_boundary_jumps(energies, mad_factor=BOUNDARY_MAD_FACTOR, n_smooth=BOUNDARY_N_SMOOTH):
    E = energies.copy().astype(float)
    N = len(E)
    if N < 4:
        return E, []

    dE = np.abs(np.diff(E))
    med = np.median(dE)
    mad = np.median(np.abs(dE - med))
    if mad < 1e-10:
        return E, []

    threshold = med + mad_factor * mad
    jump_indices = np.where(dE > threshold)[0]
    if len(jump_indices) == 0:
        return E, []

    groups = [[jump_indices[0]]]
    for idx in jump_indices[1:]:
        if idx <= groups[-1][-1] + 2:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    jump_positions = []
    for grp in groups:
        left_end = grp[0]
        right_start = grp[-1] + 1
        left_slice = E[max(0, left_end - n_smooth): left_end]
        right_slice = E[right_start + 1: min(N, right_start + 1 + n_smooth)]
        E_left = float(np.mean(left_slice)) if len(left_slice) > 0 else float(E[left_end])
        E_right = float(np.mean(right_slice)) if len(right_slice) > 0 else float(E[right_start])
        n_interp = right_start - left_end
        for j, k in enumerate(range(left_end, right_start)):
            alpha = (j + 1) / (n_interp + 1)
            E[k] = E_left * (1.0 - alpha) + E_right * alpha
        jump_positions.append((left_end, right_start))

    return E, jump_positions


# ====================================================================
# 3. Descriptor computation (numerics match the reference implementation; only the returned fields are trimmed)
# ====================================================================
def compute_descriptors(coords: np.ndarray, energies: np.ndarray,
                         axis_name: str, meta: dict | None = None) -> dict:
    """Compute the core descriptor subset (7 path-geometry fields) for a single MEP path."""
    if meta is None:
        meta = {}

    diffs_raw = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    dup_mask = np.concatenate([[True], diffs_raw > 1e-8])
    if not np.all(dup_mask):
        coords = coords[dup_mask]
        energies = energies[dup_mask]

    energies_smooth, _ = smooth_boundary_jumps(energies)

    N = len(coords)
    if N < 4:
        return {}

    ax_i = {"X": 0, "Y": 1, "Z": 2}[axis_name]

    diffs = np.diff(coords, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.], np.cumsum(ds)])
    L = float(s[-1])

    disp = coords[-1] - coords[0]
    L_dir = float(np.linalg.norm(disp))
    L_main = float(abs(disp[ax_i]))

    w = ds / ds.sum()
    E_mid = 0.5 * (energies[:-1] + energies[1:])

    E_min = float(energies.min())
    E_max = float(energies.max())
    E_a = float(E_max - E_min)
    E_mean_w = float(np.dot(w, E_mid))
    E_std_w = float(np.sqrt(np.dot(w, (E_mid - E_mean_w) ** 2)))

    peaks, _ = find_peaks(energies_smooth, prominence=3, distance=2)
    valleys, _ = find_peaks(-energies_smooth, prominence=3, distance=2)
    n_b_total = int(len(peaks))

    f_b = float(n_b_total / max(L, 1e-6))  # barrier_freq

    slope_list = []
    for p in peaks:
        lv = valleys[valleys < p]
        rv = valleys[valleys > p]
        lv_idx = int(lv[-1]) if len(lv) else 0
        rv_idx = int(rv[0]) if len(rv) else N - 1
        e_l = float(energies[lv_idx])
        e_r = float(energies[rv_idx])
        d_l = float(np.linalg.norm(coords[p] - coords[lv_idx]))
        d_r = float(np.linalg.norm(coords[rv_idx] - coords[p]))
        sl_l = (float(energies[p]) - e_l) / max(d_l, 1e-6)
        sl_r = (float(energies[p]) - e_r) / max(d_r, 1e-6)
        slope_list.append(max(sl_l, sl_r))
    peak_slope_mean = float(np.mean(slope_list)) if slope_list else 0.0

    tortuosity = float(L / max(L_dir, 1e-6))
    eta_main = float(L_main / max(L, 1e-6))

    if E_a > 0.1:
        bins = min(20, max(5, N // 5))
        hist, edges = np.histogram(energies, bins=bins, density=True)
        hist = hist[hist > 0]
        bw = np.diff(edges)[0]
        H_E = float(-np.sum(hist * np.log(hist + 1e-30)) * bw)
    else:
        H_E = 0.0

    return {
        "E_max": E_max,
        "sigma_E": E_std_w,
        "barrier_freq": f_b,
        "peak_slope_mean": peak_slope_mean,
        "tortuosity": tortuosity,
        "eta_main": eta_main,
        "H_entropy": H_E,
    }


# ====================================================================
# 4. Per-zeolite processing
# ====================================================================
def process_zeolite(folder: str) -> dict | None:
    """
    Process a single zeolite's output subfolder (the output directory
    produced by main.py / ZeoliteWorkflow). Returns one row of data
    organized by axis (X_/Y_/Z_ prefixes), containing only the 9 core descriptors.
    """
    name = os.path.basename(folder)
    if not os.path.isdir(folder):
        return None

    barriers_info = read_barriers_file(folder, name)
    summary = read_summary(folder, name)

    traversing_axes = [ax for ax in ("X", "Y", "Z") if barriers_info[ax]["traverses"]]
    dim_str = summary.get("Dimensionality") or f"{len(traversing_axes)}D"

    row = {
        "Zeolite": name,
        "Dimensionality": dim_str,
        "a_A": summary.get("a_A", float("nan")),
        "b_A": summary.get("b_A", float("nan")),
        "c_A": summary.get("c_A", float("nan")),
    }

    found_path = False
    for ax in ("X", "Y", "Z"):
        b_info = barriers_info[ax]
        traverses = b_info["traverses"]
        row[f"{ax}_traverses"] = traverses

        if traverses:
            row[f"{ax}_barrier_official"] = b_info.get("barrier", float("nan"))
            t_star = b_info.get("transition_energy", float("nan"))
            barrier = b_info.get("barrier", float("nan"))
            row[f"{ax}_E_min_official"] = (float(t_star) - float(barrier)
                                            if not (np.isnan(t_star) or np.isnan(barrier))
                                            else float("nan"))
        else:
            row[f"{ax}_barrier_official"] = float("nan")
            row[f"{ax}_E_min_official"] = float("nan")

        fpath = os.path.join(folder, f"{name}_path_{ax}.txt")
        if not os.path.exists(fpath):
            continue

        coords, energies, meta = read_path_file(fpath)
        if coords is None:
            continue

        found_path = True
        desc = compute_descriptors(coords, energies, ax, meta)
        for k, v in desc.items():
            row[f"{ax}_{k}"] = v

    if not found_path and not traversing_axes:
        return None

    return row


def expand_to_direction_rows(records: list) -> list:
    """
    Expand each zeolite's multi-axis descriptors into multiple rows (one
    per traversing direction), keeping a Framework column (the original
    zeolite name without the _x/_y/_z suffix) for the downstream merge
    with external descriptors (FDSi/Vacc/ASA) and PLD.
    """
    expanded = []
    for rec in records:
        base = {
            "Framework": rec["Zeolite"],
            "Dimensionality": rec.get("Dimensionality", "?"),
            "Direction": None,
            "a_A": rec.get("a_A", float("nan")),
            "b_A": rec.get("b_A", float("nan")),
            "c_A": rec.get("c_A", float("nan")),
        }
        for ax in ("X", "Y", "Z"):
            if not rec.get(f"{ax}_traverses", False):
                continue
            row = dict(base)
            row["Zeolite"] = f"{rec['Zeolite']}_{ax.lower()}"
            row["Direction"] = ax
            for key, val in rec.items():
                if key.startswith(f"{ax}_"):
                    desc_key = key[len(ax) + 1:]
                    row[desc_key] = val
            expanded.append(row)
    return expanded


def extract_descriptors_batch(analysis_root: str, log=print) -> list:
    """Scan all zeolite subfolders under analysis_root and return the expanded per-direction row list."""
    records = []
    subfolders = sorted(
        d for d in os.listdir(analysis_root)
        if os.path.isdir(os.path.join(analysis_root, d))
    )
    for d in subfolders:
        row = process_zeolite(os.path.join(analysis_root, d))
        if row is not None:
            records.append(row)
            log(f"  {d}: descriptor extraction complete")
        else:
            log(f"  {d}: skipped (no traversing direction or path file)")
    return expand_to_direction_rows(records)
