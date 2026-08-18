"""Per-structure and batch workflow for the energy grid, channel dimensionality, and diffusion barriers."""
import os
import io
import copy
import time
import glob
import contextlib
import traceback

import numpy as np
import pandas as pd

from .grid_generation import GridGenerator
from .pore_channel_analyzer import PoreChannelAnalyzer, CalculationError

_DEFAULTS = {
    'grid': {
        'spacing': 0.2,
        'cutoff': 14.0,
        'probe_type': 'ch4',
        'min_supercell_length': 28.0,
    },
    'barrier': {
        'energy_threshold': 300.0,
        'min_path_nodes': 5,
        'boundary_fraction': 0.10,
        'dilation_iter': 1,
        'min_dim_fraction': 0.10,
    },
    'output': {
        'save_html': False,
        'save_vts': False,
        'save_grid_txt': False,
    },
}


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress the verbose internal prints of grid_generation.py / pore_channel_analyzer.py."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


class ZeoliteWorkflow:
    """Complete potential-grid + pore-channel analysis workflow for a single zeolite CIF."""

    def __init__(self, cif_file: str, output_dir: str = None):
        self.cif_file = cif_file
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(cif_file))
        os.makedirs(self.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(cif_file))[0]
        self.output_prefix = os.path.join(self.output_dir, base)
        self._cfg = copy.deepcopy(_DEFAULTS)

    def apply_config(self, cfg: dict) -> None:
        for section in ('grid', 'barrier', 'output'):
            if section in cfg:
                self._cfg[section].update(cfg[section])

    def run(self, log=print):
        name = os.path.splitext(os.path.basename(self.cif_file))[0]
        t0 = time.time()
        try:
            gc = self._cfg['grid']
            oc = self._cfg['output']
            with _suppress_stdout():
                grid_result = GridGenerator(
                    grid_spacing=gc['spacing'], cutoff=gc['cutoff'],
                    probe_type=gc['probe_type'],
                    min_supercell_length=gc['min_supercell_length'],
                ).build(
                    self.cif_file, self.output_prefix,
                    save_vts=oc['save_vts'], save_grid_txt=oc['save_grid_txt'],
                )

                bc = self._cfg['barrier']
                analyzer = PoreChannelAnalyzer(
                    energy_threshold=bc['energy_threshold'],
                    min_path_nodes=bc['min_path_nodes'],
                    boundary_fraction=bc['boundary_fraction'],
                    dilation_iter=bc['dilation_iter'],
                    min_dim_fraction=bc.get('min_dim_fraction', 0.10),
                    save_html=oc['save_html'],
                )
                dim_result, barriers = analyzer.run(
                    grid_result['unit_cell'], output_prefix=self.output_prefix,
                )

            self._save_summary(grid_result, dim_result, barriers)
            passing = [ax for ax in ('X', 'Y', 'Z') if dim_result.get(ax) == 'Yes']
            log(f"  {name}: dimensionality={dim_result.get('Dimensionality', '?')}, "
                f"traversing axes={passing or ['none']}, {time.time() - t0:.1f} s")
            return barriers

        except CalculationError as e:
            log(f"  {name}: calculation warning: {e}")
            return None
        except Exception as e:
            log(f"  {name}: error: {e}")
            traceback.print_exc()
            return None

    def _save_summary(self, grid_result, dim_result, barriers) -> None:
        fpath = f"{self.output_prefix}_analysis_summary.txt"
        scale = grid_result['scale_factors']
        cell_A = np.linalg.norm(grid_result['original_cell'], axis=1) * 10.0
        bc = self._cfg['barrier']

        with open(fpath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\nZeolite pore channel analysis\n" + "=" * 70 + "\n\n")
            f.write("1. Supercell\n" + "-" * 70 + "\n")
            f.write(f"   Supercell factors: {scale}\n")
            f.write(f"   Original unit cell: a={cell_A[0]:.2f}  b={cell_A[1]:.2f}  c={cell_A[2]:.2f} A\n\n")

            f.write("2. Dimensionality (triple tiling + ndimage.label)\n" + "-" * 70 + "\n")
            f.write(f"   Dimensionality: {dim_result['Dimensionality']}\n")
            f.write(f"   X={dim_result['X']}  Y={dim_result['Y']}  Z={dim_result['Z']}\n\n")

            f.write(f"3. Diffusion barriers (triple-stack Dijkstra, path nodes >= {bc['min_path_nodes']})\n" + "-" * 70 + "\n")
            for ax in ('X', 'Y', 'Z'):
                info = barriers[ax]
                if info['barrier'] is not None:
                    v = (f"{info['barrier']:.4f} kJ/mol  (transition energy {info['transition_energy']:.4f} kJ/mol, "
                         f"path nodes {info['n_path_nodes']})")
                else:
                    v = "not traversing / no path found"
                f.write(f"   {ax} direction: {v}\n")


def batch_run_potential(cif_dir: str, output_root: str, cfg: dict, log=print) -> pd.DataFrame:
    """Batch-run potential-grid + pore-channel analysis; each zeolite is written to output_root/{name}/."""
    cif_files = sorted(glob.glob(os.path.join(cif_dir, '*.cif')))
    if not cif_files:
        log(f"  Warning: no .cif files found in {cif_dir}")
        return pd.DataFrame()

    os.makedirs(output_root, exist_ok=True)
    log(f"  Processing {len(cif_files)} CIF file(s); output root: {output_root}")

    records = []
    t_total = time.time()

    for i, cif_file in enumerate(cif_files, 1):
        name = os.path.splitext(os.path.basename(cif_file))[0]
        out_dir = os.path.join(output_root, name)
        log(f"  [{i}/{len(cif_files)}] {name}")

        wf = ZeoliteWorkflow(cif_file=cif_file, output_dir=out_dir)
        wf.apply_config(cfg)
        barriers = wf.run(log=log)

        if barriers is not None:
            rec = {'Zeolite': name, 'Status': 'OK'}
            for ax in ('X', 'Y', 'Z'):
                info = barriers[ax]
                rec[f'Barrier_{ax} (kJ/mol)'] = round(info['barrier'], 4) if info['barrier'] is not None else None
            records.append(rec)
        else:
            records.append({'Zeolite': name, 'Status': 'FAILED',
                             'Barrier_X (kJ/mol)': None, 'Barrier_Y (kJ/mol)': None,
                             'Barrier_Z (kJ/mol)': None})

    df = pd.DataFrame(records)
    elapsed = time.time() - t_total
    n_ok = int((df['Status'] == 'OK').sum()) if len(df) else 0
    log(f"  Batch complete: {elapsed:.1f} s total, {n_ok} succeeded, {len(df) - n_ok} failed.")
    return df
