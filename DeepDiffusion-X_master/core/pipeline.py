"""End-to-end workflow from CIF input to diffusion coefficient prediction and SHAP explanation."""
import os
import math
import time
import traceback
from dataclasses import dataclass, field

import pandas as pd

from . import (symmetry, pore_size, potential_workflow, descriptors,
               feature_merge, predict, model as model_mod, shap_explain,
               shap_original, descriptor_names)

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASSETS_DIR = os.path.join(PACKAGE_ROOT, "assets")
DEFAULT_ML_DIR = os.path.join(PACKAGE_ROOT, "ml")
DEFAULT_EXTERNAL_EXCEL = os.path.join(DEFAULT_ASSETS_DIR, "external_descriptors.xlsx")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_ML_DIR, "ann_final_model.pth")
DEFAULT_SCALER_PATH = os.path.join(DEFAULT_ML_DIR, "scaler.pkl")
DEFAULT_SHAP_BACKGROUND = os.path.join(DEFAULT_ASSETS_DIR, "shap_background.xlsx")
DEFAULT_SHAP_REFERENCE_FILES = (os.path.join(DEFAULT_ASSETS_DIR, "shap_values.xlsx"),
                                os.path.join(DEFAULT_ASSETS_DIR, "shap_values_reference.xlsx"))

# Assumed unit of the 'Ds' target column the model was trained on.
# Confirmed from training code: activation = ReLU, target = ln(Ds).
# Unit is NOT verifiable from the model file itself -- adjust
# DS_UNIT_LABEL / DS_UNIT_SCALE below if the training data used a
# different convention.
DS_UNIT_LABEL = "x1e-8 m^2/s"
DS_UNIT_SCALE = 1.0  # multiply exp(ln_Ds_pred) by this factor if a conversion is needed


@dataclass
class PipelineConfig:
    grid_spacing: float = 0.2
    grid_cutoff: float = 14.0
    probe_type: str = "ch4"
    min_supercell_length: float = 28.0
    energy_threshold: float = 300.0
    min_path_nodes: int = 5
    boundary_fraction: float = 0.10
    dilation_iter: int = 1
    min_dim_fraction: float = 0.10
    pore_size_grid_n: int = 120
    symprec: float = 0.1
    log_base: float = math.e  # training target is ln(Ds) -> natural-log inversion
    activation: str = "relu"  # confirmed from FNN_Optimized.forward()
    shap_max_background: int = 50
    shap_max_explain: int = 200
    use_shap_reference: bool = True   # reuse precomputed SHAP values for known frameworks
    shap_seed: int = 0          # permutation explainer seed; fixes run-to-run variation
    shap_max_evals: int = 8000  # model evaluations per sample; higher = lower Monte Carlo error
    save_html: bool = True

    def to_grid_barrier_cfg(self) -> dict:
        return {
            'grid': {
                'spacing': self.grid_spacing, 'cutoff': self.grid_cutoff,
                'probe_type': self.probe_type,
                'min_supercell_length': self.min_supercell_length,
            },
            'barrier': {
                'energy_threshold': self.energy_threshold,
                'min_path_nodes': self.min_path_nodes,
                'boundary_fraction': self.boundary_fraction,
                'dilation_iter': self.dilation_iter,
                'min_dim_fraction': self.min_dim_fraction,
            },
            'output': {'save_html': self.save_html, 'save_vts': False, 'save_grid_txt': False},
        }


@dataclass
class PipelineResult:
    feature_table: pd.DataFrame = None
    prediction_table: pd.DataFrame = None
    prediction_display: pd.DataFrame = None
    shap_values: object = None
    shap_background_source: str = None
    excel_path: str = None
    work_dir: str = None
    symmetry_results: list = field(default_factory=list)
    pore_size_results: list = field(default_factory=list)
    potential_summary: pd.DataFrame = None
    html_paths: list = field(default_factory=list)
    shap_sample_names: list = field(default_factory=list)
    shap_original_df: pd.DataFrame = None
    default_waterfall_path: str = None


def run_full_pipeline(cif_dir: str, work_dir: str,
                       external_excel_path: str = None,
                       model_path: str = None, scaler_path: str = None,
                       cfg: PipelineConfig = None, external_sheet_name=0,
                       skip_symmetry: bool = False,
                       log=print, stage_cb=None) -> PipelineResult:
    """
    Run the complete pipeline.

    Parameters
    ----------
    cif_dir : directory containing input CIF files
    work_dir : root directory for all intermediate and final output
    external_excel_path, model_path, scaler_path : optional overrides;
        default to the bundled assets/ml folders when None
    skip_symmetry : skip P1 symmetry expansion if CIFs are already P1
    stage_cb : optional stage_cb(name: str, fraction: float) for a GUI progress bar
    """
    external_excel_path = external_excel_path or DEFAULT_EXTERNAL_EXCEL
    model_path = model_path or DEFAULT_MODEL_PATH
    scaler_path = scaler_path or DEFAULT_SCALER_PATH

    for label, p in [("External descriptor Excel", external_excel_path),
                      ("ANN model", model_path), ("Scaler", scaler_path)]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    cfg = cfg or PipelineConfig()
    os.makedirs(work_dir, exist_ok=True)

    def _stage(name, frac):
        log(f"\n[Stage {name}]")
        if stage_cb:
            stage_cb(name, frac)

    result = PipelineResult(work_dir=work_dir)
    t0 = time.time()

    # -- 1. Symmetry removal -------------------------------------
    _stage("1/6  Symmetry removal (P1 expansion)", 0.05)
    p1_dir = os.path.join(work_dir, "01_p1_cif")
    if skip_symmetry:
        p1_dir = cif_dir
        log("  Skipped: input CIFs assumed to already be P1.")
    else:
        result.symmetry_results = symmetry.batch_remove_symmetry(cif_dir, p1_dir, symprec=cfg.symprec, log=log)

    # -- 2. Pore size analysis (GCD / directional PLD) -----------
    _stage("2/6  Pore size analysis (GCD / directional PLD)", 0.20)
    result.pore_size_results = pore_size.batch_analyze_pore_size(p1_dir, grid_n=cfg.pore_size_grid_n, log=log)

    # -- 3. Potential grid + channel dimensionality + barrier ----
    _stage("3/6  Potential grid, dimensionality, Dijkstra barrier", 0.45)
    barrier_root = os.path.join(work_dir, "02_barrier_analysis")
    result.potential_summary = potential_workflow.batch_run_potential(
        p1_dir, barrier_root, cfg.to_grid_barrier_cfg(), log=log
    )

    # -- 4. Descriptor extraction (9 core descriptors) -----------
    _stage("4/6  Descriptor extraction", 0.65)
    direction_rows = descriptors.extract_descriptors_batch(barrier_root, log=log)

    # Intermediate grid, barrier and path files are only required for
    # descriptor extraction; once extracted, the barrier analysis directory
    # retains the minimum energy path visualizations alone.
    result.html_paths = cleanup_barrier_outputs(barrier_root, log=log)

    # -- 5. Merge PLD + external Excel -> 13-feature table --------
    _stage("5/6  Merge PLD and external descriptors (FDSi/Vacc/ASA)", 0.75)
    feature_table = feature_merge.build_feature_table(
        direction_rows, result.pore_size_results, external_excel_path,
        sheet_name=external_sheet_name, log=log,
    )
    result.feature_table = feature_table

    # -- 6. ANN inference + SHAP explanation ----------------------
    _stage("6/6  ANN inference and SHAP explanation", 0.85)
    ann_model, feature_names, hp = model_mod.load_ann_checkpoint(
        model_path, activation=cfg.activation, log=log)
    scaler = model_mod.load_scaler(scaler_path)
    log(f"  Feature order (symbols):       "
        f"{[descriptor_names.symbol(f) for f in feature_names]}")
    log(f"  Feature order (internal keys): {feature_names}")

    pred_table = predict.predict_diffusion(feature_table, ann_model, scaler,
                                            feature_order=feature_names,
                                            log_base=cfg.log_base,
                                            ds_scale=DS_UNIT_SCALE, log=log)
    result.prediction_table = pred_table

    disp_cols = [c for c in ("Zeolite", "Direction") if c in pred_table.columns] + ["Ds_pred"]
    display_df = pred_table.loc[pred_table["pred_ok"], disp_cols].reset_index(drop=True)
    display_df = display_df.rename(columns={"Ds_pred": f"Ds_pred ({DS_UNIT_LABEL})"})
    result.prediction_display = display_df

    log(f"\n  Predicted diffusion coefficients ({DS_UNIT_LABEL}):")
    for _, row in display_df.iterrows():
        zname = row.get("Zeolite", "")
        val = row[f"Ds_pred ({DS_UNIT_LABEL})"]
        log(f"    {zname:<14s} Ds = {val:.4e}")

    X, valid_mask = predict.prepare_feature_matrix(feature_table, feature_names)
    if X.shape[0] > 0:
        X_scaled = scaler.transform(X)
        valid_zeolites = feature_table.loc[valid_mask, "Zeolite"].reset_index(drop=True)
        try:
            background = None
            if os.path.exists(DEFAULT_SHAP_BACKGROUND):
                background = shap_explain.load_background_from_excel(
                    DEFAULT_SHAP_BACKGROUND, feature_names, scaler, log=log,
                )
                result.shap_background_source = DEFAULT_SHAP_BACKGROUND
            else:
                log(f"  No background file at {DEFAULT_SHAP_BACKGROUND}; "
                    f"falling back to the current batch (see note below).")

            reference = {}
            ref_path = next((p for p in DEFAULT_SHAP_REFERENCE_FILES if os.path.exists(p)), None)
            if cfg.use_shap_reference and ref_path:
                try:
                    base_bg = None
                    if background is not None:
                        base_bg = float(shap_explain.background_base_value(ann_model, background))
                    reference = shap_explain.load_reference_shap(
                        ref_path, feature_names, default_base=base_bg, log=log)
                except Exception as exc:
                    log(f"  Reference SHAP file ignored: {exc}")

            shap_values, explained_idx = shap_explain.compute_shap_values_hybrid(
                ann_model, X_scaled, valid_zeolites.tolist(), reference,
                feature_names=feature_names,
                background=background, max_explain=cfg.shap_max_explain,
                seed=cfg.shap_seed, max_evals=cfg.shap_max_evals, log=log,
            )
            result.shap_values = shap_values
            result.shap_sample_names = valid_zeolites.iloc[explained_idx].reset_index(drop=True).tolist()

            shap_dir = os.path.join(work_dir, "03_shap")

            # ln(Ds) -> original-scale SHAP contribution conversion
            result.shap_original_df = shap_original.convert_all(
                shap_values, sample_names=result.shap_sample_names,
                feature_names=feature_names, log=log,
            )

            # Default waterfall plot for the first explained sample
            if result.shap_sample_names:
                default_sample = result.shap_sample_names[0]
                try:
                    result.default_waterfall_path = shap_original.waterfall_for_sample(
                        result.shap_original_df, default_sample,
                        feature_table=feature_table, feature_names=feature_names,
                        output_dir=shap_dir, log=log,
                    )
                except Exception as exc:
                    log(f"  Warning: default waterfall plot could not be generated: {exc}")
        except Exception as exc:
            log(f"  SHAP analysis failed: {exc}")
            traceback.print_exc()
    else:
        log("  No valid samples available; SHAP analysis skipped.")

    # -- Final Excel output ----------------------------------------
    excel_path = os.path.join(work_dir, "DeepDiffusionX_results.xlsx")
    write_final_excel(result, excel_path, feature_names, log=log)
    result.excel_path = excel_path

    log(f"\nPipeline completed in {time.time() - t0:.1f} s.")
    if stage_cb:
        stage_cb("Done", 1.0)
    return result


PATH_HTML_SUFFIXES = tuple(f"_diffusion_path_{ax}.html" for ax in ("X", "Y", "Z"))


def cleanup_barrier_outputs(barrier_root: str, log=print) -> list:
    """
    Remove every intermediate file under barrier_root, keeping only the
    *_diffusion_path_X/Y/Z.html visualizations. Returns the retained paths.
    """
    kept, removed = [], 0
    if not os.path.isdir(barrier_root):
        return kept
    for dirpath, _dirnames, filenames in os.walk(barrier_root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.endswith(PATH_HTML_SUFFIXES):
                kept.append(fpath)
                continue
            try:
                os.remove(fpath)
                removed += 1
            except OSError as exc:
                log(f"  Warning: could not remove {fpath}: {exc}")
    kept = sorted(kept)
    log(f"  Barrier analysis directory cleaned: {removed} intermediate file(s) removed, "
        f"{len(kept)} pathway visualization(s) retained.")
    return kept


def descriptor_display_table(feature_table: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Descriptor table as shown in the interface: identifier columns followed by descriptor symbols."""
    id_cols = [c for c in ("Zeolite", "Framework", "Direction", "Dimensionality")
               if c in feature_table.columns]
    desc_cols = [c for c in feature_names if c in feature_table.columns]
    out = feature_table[id_cols + desc_cols].copy()
    return out.rename(columns={c: descriptor_names.symbol(c) for c in desc_cols})


def _typeset_descriptor_sheet(worksheet, n_id_cols: int) -> None:
    """Set the descriptor sheet in Times New Roman, descriptor symbols in italic."""
    from openpyxl.styles import Font
    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Times New Roman", size=11)
    for cell in worksheet[1]:
        cell.font = Font(name="Times New Roman", size=11, bold=True,
                         italic=cell.column > n_id_cols)


def write_final_excel(result: PipelineResult, out_path: str, feature_names, log=print) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if result.feature_table is not None:
            desc_df = descriptor_display_table(result.feature_table, feature_names)
            desc_df.to_excel(writer, sheet_name="Descriptors", index=False)
            n_desc = sum(1 for c in feature_names if c in result.feature_table.columns)
            _typeset_descriptor_sheet(writer.sheets["Descriptors"], len(desc_df.columns) - n_desc)
        if result.prediction_display is not None:
            result.prediction_display.to_excel(writer, sheet_name="Diffusivity", index=False)
        if result.shap_original_df is not None:
            result.shap_original_df.to_excel(writer, sheet_name="SHAP_Values", index=False)

        # Nomenclature sheet: symbol to internal-key correspondence, so the
        # workbook is self-documenting.
        nomenclature = pd.DataFrame([
            {"Symbol": d.symbol, "ASCII": d.ascii, "Descriptor": d.name,
             "Unit": d.unit or "dimensionless", "Family": d.family,
             "Definition": d.definition, "Internal key": d.key}
            for d in descriptor_names.DESCRIPTORS.values()
        ])
        nomenclature.to_excel(writer, sheet_name="Nomenclature", index=False)
    log(f"  Final results saved: {out_path}")
