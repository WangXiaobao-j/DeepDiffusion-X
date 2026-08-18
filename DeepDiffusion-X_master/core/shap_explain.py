"""SHAP value computation for the trained ANN model."""
import os

import numpy as np
import pandas as pd
import torch
import shap

from .feature_merge import FEATURE_ORDER
from . import descriptor_names


def _make_predict_fn(model, device="cpu"):
    def predict_fn(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)
            return model(xt).cpu().numpy().ravel()
    return predict_fn


def resolve_feature_columns(df: pd.DataFrame, feature_names, log=print) -> pd.DataFrame:
    """
    Rename descriptor columns given as display symbols back to their internal
    keys. Columns already named by internal key take precedence.
    """
    rename = {}
    for col in df.columns:
        if col in feature_names:
            continue
        key = descriptor_names.resolve(col)
        if key and key in feature_names and key not in df.columns:
            rename[col] = key
    if rename:
        df = df.rename(columns=rename)
        log("  Background column names resolved to internal keys: "
            + ", ".join(f"{k} -> {v}" for k, v in rename.items()))
    return df


def load_background_from_excel(path: str, feature_names, scaler, log=print) -> np.ndarray:
    """
    Load a representative background feature sample (raw, unscaled
    values matching training-time descriptor units) and apply
    scaler.transform(). Columns may be named by internal key or by display
    symbol; any other column (e.g. Zeolites, Ds) is ignored.
    """
    feature_names = feature_names or FEATURE_ORDER
    df = pd.read_excel(path)
    df = resolve_feature_columns(df, feature_names, log=log)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Background file is missing required column(s): {missing}")
    sub = df[feature_names].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        raise ValueError(f"Background file {path} contains no valid numeric rows.")
    X = sub.to_numpy(dtype=float)
    log(f"  Background dataset loaded: {X.shape[0]} sample(s) from {path}")
    return scaler.transform(X)


def build_background_from_batch(X_scaled: np.ndarray, max_background: int = 50, log=print) -> np.ndarray:
    """
    Fall back: build the background from the current batch itself
    (kmeans-summarized if large). Any feature constant across the
    current batch receives zero SHAP contribution, because the
    background offers no alternative value to contrast against.
    """
    n = X_scaled.shape[0]
    log(f"  No training-reference background file found; using the current batch "
        f"({n} sample(s)) as the SHAP background. Features with no variance in "
        f"this batch (e.g. framework-level descriptors for a single zeolite) "
        f"will show zero contribution. Provide assets/shap_background.xlsx for "
        f"meaningful single-sample explanations.")
    if n <= max_background:
        return X_scaled
    log(f"  Background sample count {n} > {max_background}; summarized to {max_background} representative points via shap.kmeans.")
    return shap.kmeans(X_scaled, max_background).data


def load_reference_shap(path: str, feature_names=None, default_base: float = None,
                        log=print) -> dict:
    """
    Load precomputed ln(Ds)-scale SHAP values exported from the training run.

    The file must contain a zeolite name column and one column per feature
    (named by internal key or display symbol). A base_value column is used when
    present; otherwise default_base is applied, which should be the base value
    of the background dataset the values were computed against.

    Returns a mapping from the normalized sample name to (values, base_value).
    """
    feature_names = feature_names or FEATURE_ORDER
    df = pd.read_excel(path)
    df = resolve_feature_columns(df, feature_names, log=lambda m: None)

    name_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("zeolites", "zeolite", "name")), None)
    missing = [f for f in feature_names if f not in df.columns]
    if name_col is None or missing:
        raise ValueError("Reference SHAP file is not usable "
                         f"(name column: {name_col}, missing feature column(s): {missing}).")

    base_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("base_value", "base", "expected_value")), None)
    if base_col is None:
        if default_base is None:
            raise ValueError("Reference SHAP file has no base_value column and no "
                             "background base value is available.")
        log(f"  Reference SHAP file has no base_value column; "
            f"using the background base value {default_base:.6f}.")

    table = {}
    for _, row in df.iterrows():
        key = str(row[name_col]).strip().lower()
        base = float(row[base_col]) if base_col is not None else float(default_base)
        table[key] = (row[feature_names].to_numpy(dtype=float), base)
    log(f"  Reference SHAP values available for {len(table)} sample(s) "
        f"({os.path.basename(path)}).")
    return table


def compute_shap_values(model, X_scaled: np.ndarray, feature_names=None,
                         background: np.ndarray = None, max_explain: int = 200,
                         device="cpu", seed: int = 0, max_evals: int = 8000,
                         log=print):
    """
    Compute SHAP values.

    Parameters
    ----------
    X_scaled   : feature matrix after scaler.transform() (same feature order as training)
    background : optional background dataset (also scaled); when omitted,
                 falls back to build_background_from_batch(X_scaled)
    seed       : random seed of the permutation explainer; the permutation
                 order is otherwise drawn from the global numpy state, which
                 makes the values differ between runs
    max_evals  : model evaluations per explained sample; the number of
                 permutations averaged is max_evals // (2 * n_features + 1),
                 so larger values reduce the Monte Carlo error of the estimate

    Returns
    -------
    shap_values   : shap.Explanation (values/base_values are in the raw
                    model output scale, i.e. ln(Ds) in this project)
    explained_idx : ndarray of row indices into X_scaled that were
                    actually explained, used to map each shap_values
                    row back to its original sample (e.g. Zeolite
                    name); equals np.arange(X_scaled.shape[0]) when no
                    subsampling occurred
    """
    feature_names = feature_names or FEATURE_ORDER
    predict_fn = _make_predict_fn(model, device)

    if background is None:
        background = build_background_from_batch(X_scaled, log=log)

    explain_set = X_scaled
    explained_idx = np.arange(X_scaled.shape[0])
    if explain_set.shape[0] > max_explain:
        log(f"  Explained sample count {explain_set.shape[0]} > {max_explain}; randomly subsampled to {max_explain} for SHAP computation.")
        explained_idx = np.random.default_rng(0).choice(explain_set.shape[0], max_explain, replace=False)
        explain_set = explain_set[explained_idx]

    log(f"  Computing SHAP values (background {background.shape[0]} / explained {explain_set.shape[0]})...")
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names, seed=seed)
    shap_values = explainer(explain_set, max_evals=max_evals)
    log("  SHAP value computation complete.")
    return shap_values, explained_idx


def background_base_value(model, background: np.ndarray, device="cpu") -> float:
    """Mean model output over the background set, i.e. the SHAP base value."""
    return float(np.mean(_make_predict_fn(model, device)(background)))


def compute_shap_values_hybrid(model, X_scaled: np.ndarray, sample_names,
                                reference: dict = None, feature_names=None,
                                background: np.ndarray = None, max_explain: int = 200,
                                device="cpu", seed: int = 0, max_evals: int = 8000,
                                log=print):
    """
    Return SHAP values for every sample, reusing precomputed reference values
    where the sample name is found and computing the remainder with the
    explainer.

    Reusing the reference table keeps the values identical to those of the
    training run for frameworks it covers; new frameworks are explained against
    the same background, so their base value is the same.
    """
    feature_names = feature_names or FEATURE_ORDER
    reference = reference or {}
    names = [str(n).strip().lower() for n in sample_names]

    hit = [i for i, n in enumerate(names) if n in reference]
    miss = [i for i, n in enumerate(names) if n not in reference]
    if reference:
        log(f"  Reference SHAP values reused for {len(hit)} sample(s); "
            f"{len(miss)} sample(s) to compute.")

    if not hit:
        return compute_shap_values(model, X_scaled, feature_names=feature_names,
                                   background=background, max_explain=max_explain,
                                   device=device, seed=seed, max_evals=max_evals, log=log)

    n_feat = len(feature_names)
    values = np.zeros((len(names), n_feat), dtype=float)
    base_values = np.zeros(len(names), dtype=float)
    for i in hit:
        values[i], base_values[i] = reference[names[i]]

    if miss:
        sub, sub_idx = compute_shap_values(
            model, X_scaled[miss], feature_names=feature_names, background=background,
            max_explain=max_explain, device=device, seed=seed, max_evals=max_evals, log=log)
        sub_values = np.asarray(sub.values)
        sub_base = np.asarray(sub.base_values).ravel()
        for k, j in enumerate(sub_idx):
            values[miss[j]] = sub_values[k]
            base_values[miss[j]] = sub_base[k]

    explained_idx = np.array(sorted(hit + [miss[j] for j in range(len(miss))]))
    explanation = shap.Explanation(values=values, base_values=base_values,
                                    data=X_scaled, feature_names=list(feature_names))
    log("  SHAP value assembly complete.")
    return explanation, explained_idx


def shap_values_to_frame(shap_values, feature_names=None):
    """Convert SHAP values to a DataFrame (for writing to Excel)."""
    feature_names = feature_names or FEATURE_ORDER
    df = pd.DataFrame(shap_values.values, columns=feature_names)
    df.insert(0, "base_value", shap_values.base_values)
    return df
