"""Feature matrix preparation, scaling, and ANN inference of the diffusion coefficient."""
import math
import numpy as np
import pandas as pd
import torch

from .feature_merge import FEATURE_ORDER


def prepare_feature_matrix(df: pd.DataFrame, feature_order=None) -> tuple:
    """
    Extract the numeric matrix used for inference from the merged feature table.

    Returns
    -------
    X          : (N, n_features) ndarray, rows with NaN excluded
    valid_mask : boolean array (length = len(df)) marking which rows made it into X
    """
    feature_order = feature_order or FEATURE_ORDER
    sub = df[feature_order].apply(pd.to_numeric, errors="coerce")
    valid_mask = ~sub.isna().any(axis=1)
    X = sub.loc[valid_mask].to_numpy(dtype=float)
    return X, valid_mask.to_numpy()


def predict_diffusion(df: pd.DataFrame, model, scaler, feature_order=None,
                       log_base: float = math.e, ds_scale: float = 1.0,
                       log=print) -> pd.DataFrame:
    """
    Run scaling + ANN inference on the merged feature table.

    Returns
    -------
    A copy of df with new columns:
        ln_Ds_pred - raw model output (ln(Ds), consistent with the training
                     target np.log(target))
        Ds_pred    - exp(ln_Ds_pred) * ds_scale (log_base defaults to e;
                     ds_scale applies any unit conversion, default 1.0)
        pred_ok    - whether the row was successfully predicted (False = skipped due to missing features)
    """
    feature_order = feature_order or FEATURE_ORDER
    out = df.copy()
    out["ln_Ds_pred"] = np.nan
    out["Ds_pred"] = np.nan
    out["pred_ok"] = False

    X, valid_mask = prepare_feature_matrix(df, feature_order)
    n_skipped = int((~valid_mask).sum())
    if n_skipped:
        log(f"  {n_skipped} row(s) skipped due to missing feature values.")
    if X.shape[0] == 0:
        log("  No complete samples available for prediction.")
        return out

    X_scaled = scaler.transform(X)
    with torch.no_grad():
        xt = torch.as_tensor(X_scaled, dtype=torch.float32)
        y_pred = model(xt).cpu().numpy().ravel()

    ds_values = (np.power(log_base, y_pred) if log_base != math.e else np.exp(y_pred)) * ds_scale

    out.loc[valid_mask, "ln_Ds_pred"] = y_pred
    out.loc[valid_mask, "Ds_pred"] = ds_values
    out.loc[valid_mask, "pred_ok"] = True

    log(f"  Diffusion coefficient prediction completed for {X.shape[0]} sample(s).")
    return out
