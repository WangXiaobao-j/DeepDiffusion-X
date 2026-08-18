"""Global cavity diameter (GCD) and directional pore limiting diameter (PLD) from a voxel free-radius grid."""
import time
from itertools import product
from pathlib import Path

import numpy as np

# -- Atomic radii (A) ------------------------------------------------------
# Pure-silica convention; the radii enter the free-radius field directly, so
# any change shifts every PLD systematically.
ATOM_RADII = {
    "Si": 1.266,
    "O": 1.266,
}
DEFAULT_RADIUS = 1.35

GRID_N_DEFAULT = 120     # voxels per lattice direction
N_SHELL_DEFAULT = 1      # +/-1 cell shell, so PBC neighbours enter the KD-tree
PROBE_R_DEFAULT = 0.0    # free-radius floor; 0.0 = hard-sphere free volume

_ORTHOGONALITY_TOL = 1e-6


def _parse_cif(path):
    from pymatgen.io.cif import CifParser
    return CifParser(str(path)).get_structures(primitive=False)[0]


def _build_supercell(struct, radii_arr, n=1):
    """Replicate the cell into a (2n+1)^3 block so periodic images are searchable."""
    lat, frac = struct.lattice, struct.frac_coords
    carts, rads = [], []
    for cell in product(range(-n, n + 1), repeat=3):
        off = np.array(cell, dtype=float)
        for f, r in zip(frac, radii_arr):
            carts.append(lat.get_cartesian_coords(f + off))
            rads.append(r)
    return np.array(carts), np.array(rads)


def build_free_r_grid(struct, radii_dict, grid_n=GRID_N_DEFAULT, n_shell=N_SHELL_DEFAULT):
    """
    Free-radius field: for every voxel centre, the distance to the nearest atom
    surface (nearest-neighbour distance minus that atom's radius). Negative
    inside an atom, positive in open space.

    Voxel centres are offset by half a voxel ((i + 0.5) / grid_n) so that no
    sample point lands exactly on an atomic surface, which would otherwise make
    the >= comparison in the bisection sensitive to floating-point noise.
    """
    from scipy.spatial import cKDTree
    lat = struct.lattice
    L = lat.matrix
    radii_arr = np.array([radii_dict.get(s.specie.symbol, DEFAULT_RADIUS) for s in struct])
    cart_sc, atom_rad_sc = _build_supercell(struct, radii_arr, n_shell)
    tree = cKDTree(cart_sc)
    t = (np.arange(grid_n) + 0.5) / grid_n
    fa, fb, fc = np.meshgrid(t, t, t, indexing="ij")
    grid_frac = np.column_stack([fa.ravel(), fb.ravel(), fc.ravel()])
    grid_cart = grid_frac @ L
    dists, nn = tree.query(grid_cart, k=1)
    free_r = (dists - atom_rad_sc[nn]).reshape(grid_n, grid_n, grid_n)
    return free_r


def _percolates(accessible, direction, lattice=None):
    """
    Does the accessible voxel set percolate along ``direction`` under PBC?

    scipy.ndimage.label gives 6-connected components in the C layer; a
    union-find then merges components that meet across each *transverse*
    periodic face, and percolation holds when the entry and exit faces of the
    percolation axis share a union-find root.

    A transverse face pair is merged only if that lattice vector is orthogonal
    to the percolation direction. For monoclinic and triclinic cells the skewed
    vector does not map voxel (i, j, 0) onto voxel (i, j, n-1) in real space,
    so merging there manufactures a channel that does not exist. Passing
    ``lattice=None`` skips the check and assumes an orthogonal cell.
    """
    from scipy.ndimage import label as ndlabel
    shape = accessible.shape

    conn6 = np.zeros((3, 3, 3), dtype=bool)
    conn6[1, 1, :] = True
    conn6[1, :, 1] = True
    conn6[:, 1, 1] = True
    labeled, n_labels = ndlabel(accessible, structure=conn6)

    if n_labels == 0:
        return False

    parent = np.arange(n_labels + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path halving
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for transverse in range(3):
        if transverse == direction:
            continue
        if lattice is not None:
            dot = abs(np.dot(lattice.matrix[direction], lattice.matrix[transverse]))
            if dot > _ORTHOGONALITY_TOL:
                continue
        face_0 = labeled.take(0, axis=transverse)
        face_n1 = labeled.take(shape[transverse] - 1, axis=transverse)
        mask = (face_0 > 0) & (face_n1 > 0) & (face_0 != face_n1)
        if mask.any():
            for a, b in zip(face_0[mask], face_n1[mask]):
                union(a, b)

    entry_labels = set(labeled.take(0, axis=direction).ravel()) - {0}
    exit_labels = set(labeled.take(shape[direction] - 1, axis=direction).ravel()) - {0}

    entry_roots = {find(l) for l in entry_labels}
    exit_roots = {find(l) for l in exit_labels}

    return bool(entry_roots & exit_roots)


def pld_direction(free_r_grid, direction, lattice=None, probe_r=PROBE_R_DEFAULT):
    """
    Largest sphere diameter that percolates along ``direction``.

    Bisection over the sorted distinct free-radius values rather than over a
    continuous interval: percolation is a step function of the threshold whose
    breakpoints are exactly those values, so this converges in
    log2(n_unique) labelling passes and returns a value the grid can actually
    represent.
    """
    valid_r = np.unique(free_r_grid[free_r_grid > probe_r])
    if valid_r.size == 0:
        return 0.0
    lo, hi, best_r = 0, int(valid_r.size) - 1, 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        r = float(valid_r[mid])
        if _percolates(free_r_grid >= r, direction, lattice):
            best_r = r
            lo = mid + 1
        else:
            hi = mid - 1
    return 2.0 * best_r


# Retained under its previous name so existing call sites keep working.
get_pld_directional = pld_direction


def compute_pld_gcd(free_r_grid, lattice=None, probe_r=PROBE_R_DEFAULT):
    """Return ({'a': .., 'b': .., 'c': ..}, gcd) for a free-radius grid."""
    gcd = float(2.0 * free_r_grid.max())
    pld = {ax: pld_direction(free_r_grid, d, lattice, probe_r)
           for d, ax in enumerate(["a", "b", "c"])}
    return pld, gcd


def analyze_pore_size(cif_path, radii_dict=None, grid_n=GRID_N_DEFAULT,
                       n_shell=N_SHELL_DEFAULT, probe_r=PROBE_R_DEFAULT) -> dict:
    """
    Compute GCD and directional PLD for a single CIF.

    Returns
    -------
    dict: {name, gcd, pld_a, pld_b, pld_c, ok, error, time_s}
    pld_a/b/c correspond to lattice vectors a/b/c and are mapped to X/Y/Z
    downstream by feature_merge._AXIS_TO_PLD_COL.
    """
    if radii_dict is None:
        radii_dict = dict(ATOM_RADII)
    cif_path = Path(cif_path)
    t0 = time.perf_counter()
    rec = dict(name=cif_path.stem, gcd=0.0, pld_a=0.0, pld_b=0.0, pld_c=0.0,
               ok=False, error="")
    try:
        struct = _parse_cif(cif_path)
        free_r = build_free_r_grid(struct, radii_dict, grid_n, n_shell)
        pld, gcd = compute_pld_gcd(free_r, struct.lattice, probe_r)

        rec.update(gcd=round(gcd, 4),
                    pld_a=round(pld["a"], 4),
                    pld_b=round(pld["b"], 4),
                    pld_c=round(pld["c"], 4),
                    ok=True)
    except Exception as exc:
        rec["error"] = str(exc)
    rec["time_s"] = round(time.perf_counter() - t0, 2)
    return rec


def batch_analyze_pore_size(cif_dir, grid_n=GRID_N_DEFAULT, log=print) -> list:
    cif_dir = Path(cif_dir)
    cif_files = sorted(set(list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.CIF"))))
    results = []
    for idx, path in enumerate(cif_files, 1):
        rec = analyze_pore_size(path, grid_n=grid_n)
        results.append(rec)
        status = "OK" if rec["ok"] else "FAILED"
        if rec["ok"]:
            detail = (f"GCD {rec['gcd']:.3f} A  |  "
                      f"PLD a/b/c {rec['pld_a']:.3f} / {rec['pld_b']:.3f} / {rec['pld_c']:.3f} A")
        else:
            detail = rec["error"]
        log(f"  {status} [{idx}/{len(cif_files)}] {rec['name']}  ({detail})")
    n_ok = sum(r["ok"] for r in results)
    log(f"  Pore size analysis complete: {n_ok}/{len(results)} succeeded.")
    return results
