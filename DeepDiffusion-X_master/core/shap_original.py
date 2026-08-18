"""Conversion of SHAP contributions from ln(Ds) to the original Ds scale, with waterfall plots."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import shap

from .feature_merge import FEATURE_ORDER
from . import descriptor_names

# Unified color scheme: ochre-red = positive contribution (raises the prediction), steel-blue = negative contribution (lowers it)
POS_COLOR = "#C0441A"
NEG_COLOR = "#2C5F8A"
NEUTRAL_DARK = "#333333"

_SHAP_NATIVE_RED = np.array(to_rgb("#FF0051"))
_SHAP_NATIVE_BLUE = np.array(to_rgb("#008BFB"))


# ====================================================================
# 1. ln(Ds)-scale SHAP -> original Ds-scale contributions
# ====================================================================
def log_shap_to_original(log_shap: np.ndarray, base: float):
    """
    Single-sample conversion: proportionally decompose the ln(Ds)-scale
    SHAP contributions into original Ds-scale contributions.

    Matches the reference script's log_to_original() exactly:
        log_pred = base + sum(log_shap)
        orig_base = exp(base)
        orig_pred = exp(log_pred)
        contrib_i = (log_shap_i / sum(log_shap)) * (orig_pred - orig_base)

    Returns
    -------
    contrib : ndarray, additive per-feature contribution in the original
              scale (sum(contrib) + orig_base ~= orig_pred)
    orig_pred, orig_base : float
    """
    log_pred = base + np.sum(log_shap)
    orig_base = np.exp(base)
    orig_pred = np.exp(log_pred)

    total = np.sum(log_shap)
    if total == 0:
        return np.zeros_like(log_shap), orig_pred, orig_base

    ratio = log_shap / total
    contrib = ratio * (orig_pred - orig_base)
    return contrib, orig_pred, orig_base


def convert_all(shap_values, sample_names=None, feature_names=None, log=print) -> pd.DataFrame:
    """
    Apply the log-to-original-scale conversion to every sample in a shap.Explanation.

    Parameters
    ----------
    shap_values  : Explanation returned by shap_explain.compute_shap_values()
                   (values and base_values both in ln(Ds) scale)
    sample_names : optional list of sample names matching shap_values in length (e.g. Zeolite)

    Returns
    -------
    DataFrame with columns: Zeolites, <feature...> (original-scale contribution), orig_pred, orig_base
    """
    feature_names = feature_names or FEATURE_ORDER
    values = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim == 0:
        base_values = np.full(values.shape[0], float(base_values))

    n = values.shape[0]
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(n)]

    rows = []
    for i in range(n):
        contrib, orig_pred, orig_base = log_shap_to_original(values[i], float(base_values[i]))
        row = {"Zeolites": sample_names[i], "orig_pred": orig_pred, "orig_base": orig_base}
        row.update({f: v for f, v in zip(feature_names, contrib)})
        rows.append(row)

    out = pd.DataFrame(rows)
    log(f"  Converted SHAP contributions to original Ds scale for {n} sample(s).")
    return out


# ====================================================================
# 2. Nature-style waterfall recoloring
# ====================================================================
def recolor_shap_waterfall(ax, pos_color=POS_COLOR, neg_color=NEG_COLOR, text_color=NEUTRAL_DARK):
    """Recolor shap.plots.waterfall's native red/blue arrows to ochre-red/steel-blue (logic matches the reference script)."""
    pos_rgb = np.array(to_rgb(pos_color))
    neg_rgb = np.array(to_rgb(neg_color))
    for patch in ax.patches:
        try:
            face = np.array(patch.get_facecolor()[:3])
        except (TypeError, IndexError):
            continue
        dist_red = np.linalg.norm(face - _SHAP_NATIVE_RED)
        dist_blue = np.linalg.norm(face - _SHAP_NATIVE_BLUE)
        if min(dist_red, dist_blue) < 0.5:
            new_rgb = pos_rgb if dist_red <= dist_blue else neg_rgb
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((*new_rgb, a))

    for txt in ax.texts:
        try:
            c = np.array(to_rgb(txt.get_color()))
        except ValueError:
            continue
        if np.allclose(c, 1.0, atol=0.05):
            continue
        dist_red = np.linalg.norm(c - _SHAP_NATIVE_RED)
        dist_blue = np.linalg.norm(c - _SHAP_NATIVE_BLUE)
        if min(dist_red, dist_blue) < 0.5:
            txt.set_color(text_color)


# ====================================================================
# 3. Single-sample, original-scale waterfall plot
# ====================================================================
def nature_waterfall(contrib: np.ndarray, orig_base: float, feature_names=None,
                      feature_values: np.ndarray = None, sample_name: str = "",
                      output_path: str = None, log=print) -> str:
    """
    Draw a single-sample, original-scale SHAP waterfall plot (Nature color scheme).

    Parameters
    ----------
    contrib        : this sample's per-feature original-scale contribution
                      (output of log_shap_to_original())
    orig_base      : original-scale baseline value, exp(base)
    feature_values : optional, this sample's true feature values (used for
                      labels such as "8.33 = tortuosity")
    output_path    : auto-generated in the current directory when None

    Returns
    -------
    path to the saved PNG
    """
    feature_names = feature_names or FEATURE_ORDER
    if output_path is None:
        safe_name = sample_name.replace("/", "_").replace("\\", "_") or "sample"
        output_path = f"waterfall_{safe_name}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Axis labels carry the bare publication symbols (Ebarrier, ρbarrier,
    # GE,mean, Pτ, HE ...) with no unit appended, matching the interface
    # tables. They are plain Unicode rather than mathtext markup, so the Greek
    # characters render identically here and in the GUI; italicisation is
    # applied to the tick labels after plotting, below.
    display_names = [descriptor_names.symbol(f) for f in feature_names]

    explanation = shap.Explanation(
        values=np.asarray(contrib, dtype=float),
        base_values=float(orig_base),
        data=feature_values,
        feature_names=display_names,
    )

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": NEUTRAL_DARK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": False,
    })

    with plt.rc_context(matplotlib.rcParams):
        shap.plots.waterfall(explanation, max_display=len(feature_names), show=False)
        fig = plt.gcf()
        ax = fig.axes[0]
        recolor_shap_waterfall(ax)
        # Journal convention: no top/right spines, hairline axes, no grid.
        for side in ("top", "right"):
            if side in ax.spines:
                ax.spines[side].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=10, colors=NEUTRAL_DARK)
        # Descriptor names sit on the y axis; set them in italic, as variable
        # names are in a manuscript. shap prefixes each with the sample's
        # feature value ("8.33 = Pτ"), so the whole label is italicised
        # together rather than the symbol alone.
        for label in ax.get_yticklabels():
            label.set_fontstyle("italic")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  Waterfall plot saved: {output_path}")
    return output_path


def waterfall_for_sample(shap_original_df: pd.DataFrame, sample_name: str,
                          feature_table: pd.DataFrame = None, feature_names=None,
                          output_dir: str = ".", log=print) -> str:
    """
    Look up a given sample in the original-scale contribution table
    produced by convert_all() and draw its waterfall plot. If
    feature_table (containing the raw, unscaled feature values) is
    supplied, that sample's true feature values are automatically
    matched for label display.

    Raises
    ------
    ValueError: sample name not found
    """
    feature_names = feature_names or FEATURE_ORDER
    match = shap_original_df[shap_original_df["Zeolites"] == sample_name]
    if match.empty:
        available = shap_original_df["Zeolites"].head(10).tolist()
        raise ValueError(f"Sample '{sample_name}' not found in SHAP results. Available samples (examples): {available}")
    row = match.iloc[0]
    contrib = row[feature_names].values.astype(float)
    orig_base = float(row["orig_base"])

    feature_values = None
    if feature_table is not None and "Zeolite" in feature_table.columns:
        desc_match = feature_table[feature_table["Zeolite"] == sample_name]
        if not desc_match.empty:
            try:
                feature_values = desc_match.iloc[0][feature_names].values.astype(float)
            except Exception:
                feature_values = None

    out_path = os.path.join(output_dir, f"waterfall_{sample_name}.png")
    return nature_waterfall(contrib, orig_base, feature_names=feature_names,
                             feature_values=feature_values, sample_name=sample_name,
                             output_path=out_path, log=log)
