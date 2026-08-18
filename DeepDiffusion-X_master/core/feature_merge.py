"""Assembly of the 13-dimensional model feature table from pathway descriptors, PLD, and external descriptors."""
import pandas as pd

# Order matches feature_names in the ann_final_model.pth checkpoint exactly
FEATURE_ORDER = [
    "barrier_official", "sigma_E", "eta_main", "tortuosity", "E_max",
    "E_min_official", "peak_slope_mean", "H_entropy", "barrier_freq",
    "PLD", "FDSi", "Vacc", "ASA",
]

# Directions with PLD below this value (probe kinetic diameter, CH4 ~3.5 A)
# are treated as non-diffusing and excluded before prediction, matching
# descriptor_calculator_v4's convention (PLD minimum 3.5 A).
PLD_MIN_THRESHOLD = 3.5

_AXIS_TO_PLD_COL = {"X": "pld_a", "Y": "pld_b", "Z": "pld_c"}


def pld_results_to_frame(pore_size_results: list) -> pd.DataFrame:
    """Convert pore_size.batch_analyze_pore_size() results into a long table (Framework, Direction, PLD)."""
    rows = []
    for rec in pore_size_results:
        if not rec.get("ok"):
            continue
        for ax, col in _AXIS_TO_PLD_COL.items():
            rows.append({"Framework": rec["name"], "Direction": ax, "PLD": rec[col]})
    return pd.DataFrame(rows)


def load_external_excel(path: str, sheet_name=0) -> pd.DataFrame:
    """
    Read the external Excel file, which must contain the columns:
    Zeolites, FDSi, Vacc, ASA. The Zeolites column is used to match
    against the Framework name (case-insensitive, whitespace-trimmed).
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"Zeolites", "FDSi", "Vacc", "ASA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"External Excel is missing required column(s): {missing} (present columns: {list(df.columns)})")
    df = df[["Zeolites", "FDSi", "Vacc", "ASA"]].copy()
    df["_match_key"] = df["Zeolites"].astype(str).str.strip().str.lower()
    return df


def build_feature_table(direction_rows: list, pore_size_results: list,
                         external_excel_path: str, sheet_name=0,
                         log=print) -> pd.DataFrame:
    """
    Main merge entry point.

    Parameters
    ----------
    direction_rows       : output of descriptors.extract_descriptors_batch()
    pore_size_results    : output of pore_size.batch_analyze_pore_size()
    external_excel_path  : path to the external Excel file (Zeolites, FDSi, Vacc, ASA)

    Returns
    -------
    DataFrame with three identifier columns (Zeolite, Framework, Direction)
    plus the 13 feature columns in FEATURE_ORDER (missing values become
    NaN and are logged as warnings).
    """
    if not direction_rows:
        raise ValueError("Descriptor extraction produced no rows; cannot build feature table.")

    df = pd.DataFrame(direction_rows)

    # -- Merge PLD ----------------------------------------------
    pld_df = pld_results_to_frame(pore_size_results)
    if pld_df.empty:
        log("  Warning: no PLD results available; PLD column will be empty.")
        df["PLD"] = float("nan")
    else:
        df = df.merge(pld_df, on=["Framework", "Direction"], how="left")
        n_missing = df["PLD"].isna().sum()
        if n_missing:
            log(f"  Warning: {n_missing} row(s) could not be matched to a PLD value (Framework/Direction not found in pore size results).")

    # -- Exclude directions with PLD below the probe kinetic diameter --
    # A pore limiting diameter smaller than the probe molecule cannot be
    # traversed regardless of what the energy-based barrier analysis found;
    # such directions are treated as non-diffusing and dropped here.
    below = df["PLD"].notna() & (df["PLD"] < PLD_MIN_THRESHOLD)
    if below.any():
        excluded = df.loc[below, ["Zeolite", "PLD"]]
        for _, r in excluded.iterrows():
            log(f"  Excluded (non-diffusing): {r['Zeolite']} -- PLD = {r['PLD']:.3f} A < {PLD_MIN_THRESHOLD} A")
        df = df.loc[~below].reset_index(drop=True)

    # -- Merge external Excel (FDSi / Vacc / ASA) -----------------
    # First match on Framework (suffix _x/_y/_z stripped); for rows
    # still unmatched, retry against the full Zeolite name (with
    # suffix), in case the external Excel keys its Zeolites column by
    # direction-suffixed names (e.g. "MFI_x").
    ext_df = load_external_excel(external_excel_path, sheet_name=sheet_name)
    ext_lookup = ext_df.set_index("_match_key")[["FDSi", "Vacc", "ASA"]]

    df["_framework_key"] = df["Framework"].astype(str).str.strip().str.lower()
    df["_zeolite_key"] = df["Zeolite"].astype(str).str.strip().str.lower()

    for col in ("FDSi", "Vacc", "ASA"):
        df[col] = df["_framework_key"].map(ext_lookup[col])
        still_missing = df[col].isna()
        if still_missing.any():
            df.loc[still_missing, col] = df.loc[still_missing, "_zeolite_key"].map(ext_lookup[col])

    df = df.drop(columns=["_framework_key", "_zeolite_key"])

    for col in ("FDSi", "Vacc", "ASA"):
        n_missing = df[col].isna().sum()
        if n_missing:
            unmatched = sorted(set(df.loc[df[col].isna(), "Framework"]))
            log(f"  Warning: {n_missing} row(s) could not be matched to {col} "
                f"(no corresponding Zeolites name in the external Excel): "
                f"{unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    # -- Ensure every FEATURE_ORDER column exists -----------------
    for col in FEATURE_ORDER:
        if col not in df.columns:
            log(f"  Warning: feature column {col} is missing; filled with NaN.")
            df[col] = float("nan")

    id_cols = ["Zeolite", "Framework", "Direction", "Dimensionality"]
    id_cols = [c for c in id_cols if c in df.columns]
    ordered_cols = id_cols + FEATURE_ORDER
    df = df[ordered_cols]

    return df
