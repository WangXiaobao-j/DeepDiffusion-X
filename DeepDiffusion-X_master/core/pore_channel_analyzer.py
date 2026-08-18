"""Channel dimensionality identification and diffusion barrier calculation on the energy grid."""

from __future__ import annotations

import os
import heapq
import numpy as np
from scipy.ndimage import label as _ndimage_label, binary_dilation
import plotly.graph_objects as go
from plotly.offline import plot


# ======================================================================
# Exceptions
# ======================================================================
class CalculationError(Exception):
    """Raised when no valid MEP is found for a direction identified as traversable."""


# ======================================================================
# Analyzer
# ======================================================================
class PoreChannelAnalyzer:
    """
    Pore channel analysis based on fractional-coordinate voxel topology.

    Parameters
    ----------
    energy_threshold : float
        Upper energy bound of the pore region, E_min + threshold (kJ/mol)
    min_path_nodes   : int
        Minimum number of nodes for a valid MEP; shorter paths are rejected
    boundary_fraction: float
        Fraction of the traversal axis assigned to the start/end regions
    dilation_iter    : int
        binary_dilation iterations used for topology repair (0 = no repair)
    """

    def __init__(
        self,
        energy_threshold:  float = 300.0,
        min_path_nodes:    int   = 5,
        boundary_fraction: float = 0.10,
        dilation_iter:     int   = 1,
        min_dim_fraction:  float = 0.10,
        save_html:         bool  = True,
    ):
        self.energy_threshold  = float(energy_threshold)
        self.min_path_nodes    = int(min_path_nodes)
        self.boundary_fraction = float(boundary_fraction)
        self.dilation_iter     = int(dilation_iter)
        # Minimum share of first-copy pore voxels that the traversing component
        # must cover; smaller shares are treated as thin spurious connections
        self.min_dim_fraction  = float(min_dim_fraction)
        # Write HTML visualizations (pore envelope and MEP path figures)
        self.save_html         = bool(save_html)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        unit_cell_grid_data: dict,
        output_prefix: str | None = None,
    ) -> tuple[dict, dict]:
        """
        Full analysis: dimensionality identification and barrier evaluation,
        with optional file output and visualization.

        Parameters
        ----------
        unit_cell_grid_data : dict
            GridGenerator.build()['unit_cell'], containing:
              grid_energies (nx,ny,nz ndarray, kJ/mol)
              cell          (3,3 ndarray, nm; rows are lattice vectors)
              n_grid        [nx, ny, nz]
        output_prefix : str | None
            Output file prefix; None skips all file output

        Returns
        -------
        dim_result : dict
            {'Dimensionality': '1D', 'X': 'Yes', 'Y': 'No', 'Z': 'No'}
        barriers : dict
            {'X': result_dict, 'Y': ..., 'Z': ...}, where result_dict holds
            barrier, transition_energy, min_energy, path_length,
            n_path_nodes, path_coords (N,3 in A), path_energies (N, kJ/mol)

        Raises
        ------
        CalculationError
            No valid MEP found for a direction identified as traversable
        """
        energy_grid: np.ndarray = unit_cell_grid_data['grid_energies']
        cell_A: np.ndarray      = unit_cell_grid_data['cell'] * 10.0  # nm -> Angstrom

        nx, ny, nz = energy_grid.shape
        lengths    = np.linalg.norm(cell_A, axis=1)
        print(
            f"  Voxel grid: {nx}x{ny}x{nz}  "
            f"|a|={lengths[0]:.2f}  |b|={lengths[1]:.2f}  |c|={lengths[2]:.2f} A"
        )

        # 1. Pore voxels
        pore_mask, e_min = self._build_pore_mask(energy_grid)
        n_pore = int(pore_mask.sum())
        pct    = 100.0 * n_pore / (nx * ny * nz)
        print(
            f"  Pore voxels: {n_pore:,}/{nx*ny*nz:,} ({pct:.1f}%)  "
            f"E_min = {e_min:.2f} kJ/mol  upper bound = {e_min + self.energy_threshold:.2f} kJ/mol"
        )
        if n_pore == 0:
            raise ValueError("No accessible pore voxels; lower energy_threshold or check the input grid.")

        # 2. Optional topology repair
        pore_topo = self._repair_topology(pore_mask)

        # 3. Dimensionality identification.
        # The original (non-dilated) pore_mask is used here: dilation can
        # create spurious connections along non-traversable directions and
        # overestimate the dimensionality. pore_topo is only a fallback for
        # the barrier search.
        traverses = self._identify_dimensionality(pore_mask)
        dim       = sum(traverses)
        ax_label  = ['X', 'Y', 'Z']
        passing   = [ax_label[i] for i in range(3) if traverses[i]]
        print(f"  Channel dimensionality: {dim}D  traversing directions: {passing or ['none']}")

        # 4. Barrier evaluation
        barriers: dict[str, dict] = {}
        failed: list[str]         = []

        for axis_idx in range(3):
            ax = ax_label[axis_idx]
            if not traverses[axis_idx]:
                barriers[ax] = _empty_result(e_min)
                continue

            print(f"\n  Barrier calculation along {ax}")
            res = self._compute_barrier(
                energy_grid, pore_mask, pore_topo, axis_idx, e_min, cell_A
            )
            barriers[ax] = res

            if res['barrier'] is not None:
                print(
                    f"    Barrier = {res['barrier']:.3f} kJ/mol  "
                    f"transition energy = {res['transition_energy']:.3f} kJ/mol  "
                    f"path nodes = {res['n_path_nodes']}"
                )
            else:
                print(f"    No valid traversing path found (path nodes = {res['n_path_nodes']})")
                failed.append(ax)

        if failed:
            raise CalculationError(
                f"No valid MEP found for traversing direction(s) {failed}; "
                "possible causes: energy_threshold too small, insufficient grid "
                "resolution, or a channel disconnected at this resolution "
                "(try increasing dilation_iter)."
            )

        # 5. Output
        if output_prefix:
            self._save_barrier_txt(barriers, output_prefix)
            self._save_path_txts(barriers, output_prefix)
            if self.save_html:
                self._visualize(energy_grid, pore_mask, cell_A,
                                barriers, e_min, output_prefix)
            else:
                print("  Skipped: HTML visualization (output.save_html = false)")

        dim_result = {
            'Dimensionality': f'{dim}D',
            'X': 'Yes' if traverses[0] else 'No',
            'Y': 'Yes' if traverses[1] else 'No',
            'Z': 'Yes' if traverses[2] else 'No',
        }
        return dim_result, barriers

    # ==================================================================
    # Pore voxel construction
    # ==================================================================
    def _build_pore_mask(
        self, energy_grid: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Energy threshold filter: keep voxels with E in [E_min, E_min + threshold].

        E_min is taken over grid points with energy < 1000 kJ/mol, which
        excludes the atomic core regions.
        """
        valid = energy_grid[energy_grid < 1000.0]
        if valid.size == 0:
            raise ValueError("All grid energies exceed 1000 kJ/mol; check the LJ parameters or the CIF structure.")
        e_min = float(valid.min())
        mask  = (energy_grid >= e_min - 1e-8) & \
                (energy_grid <= e_min + self.energy_threshold)
        return mask, e_min

    # ==================================================================
    # Topology repair
    # ==================================================================
    def _repair_topology(self, pore_mask: np.ndarray) -> np.ndarray:
        """
        Repair voxel-level disconnections at narrow necks with binary_dilation.

        Used for connectivity tests only; the barrier search prefers the
        original pore_mask so that the true energy landscape is preserved
        (dilated voxels may exceed the pore energy threshold). Returns a copy
        of the input mask when dilation_iter = 0.
        """
        if self.dilation_iter <= 0:
            return pore_mask.copy()

        struct26 = np.ones((3, 3, 3), dtype=bool)
        dilated  = binary_dilation(
            pore_mask, structure=struct26, iterations=self.dilation_iter
        )

        # Warn if dilation merged a substantial number of independent components
        s26i    = struct26.astype(int)
        n_orig  = int(_ndimage_label(pore_mask, structure=s26i)[1])
        n_dil   = int(_ndimage_label(dilated,   structure=s26i)[1])
        if n_orig > 2 and n_dil < max(1, n_orig // 2):
            print(
                f"  Warning: binary_dilation may have merged independent channels "
                f"({n_orig} -> {n_dil} connected components); "
                "set dilation_iter=0 to disable topology repair."
            )
        return dilated

    # ==================================================================
    # Dimensionality identification: triple tiling + ndimage.label
    # ==================================================================
    def _identify_dimensionality(self, pore_mask: np.ndarray) -> list[bool]:
        """
        Triple-tiling connectivity test along each direction (axis 0/1/2).

        Procedure:
          1. tile pore_mask three times along direction d (length 3*n_d)
          2. label connected components with scipy.ndimage.label (26-connectivity)
          3. collect the label sets of the first copy (d in [0, n_d)) and of
             the third copy (d in [2*n_d, 3*n_d))
          4. a non-empty intersection indicates a component crossing the
             periodic boundary, i.e. a candidate traversing direction

        Minimum traversing fraction
        ---------------------------
        A non-empty label intersection alone accepts thin incidental contacts
        near the cell boundary and systematically overestimates the
        dimensionality. The pore voxels of the first copy belonging to the
        traversing component must additionally cover at least
        min_dim_fraction of all first-copy pore voxels: for a genuinely
        traversing direction nearly all pore voxels belong to that component,
        whereas a spurious contact involves only a few boundary voxels.
        """
        struct26  = np.ones((3, 3, 3), dtype=int)
        traverses = [False, False, False]

        for d in range(3):
            n_d   = pore_mask.shape[d]
            stack = np.concatenate([pore_mask] * 3, axis=d)

            labels_arr, _ = _ndimage_label(stack, structure=struct26)

            slc_f    = [slice(None)] * 3;  slc_f[d] = slice(0, n_d)
            slc_l    = [slice(None)] * 3;  slc_l[d] = slice(2 * n_d, 3 * n_d)

            zone_f   = labels_arr[tuple(slc_f)]   # label matrix of the first copy
            labels_f = set(zone_f.ravel())         - {0}
            labels_l = set(labels_arr[tuple(slc_l)].ravel()) - {0}

            common = labels_f & labels_l
            if not common:
                continue   # empty intersection: not traversing

            # Minimum traversing fraction test
            total_pore_first = int((zone_f > 0).sum())
            if total_pore_first == 0:
                continue

            # First-copy pore voxels belonging to the traversing component
            traversing_first = int(np.isin(zone_f, list(common)).sum())
            fraction = traversing_first / total_pore_first

            if fraction >= self.min_dim_fraction:
                traverses[d] = True
            else:
                print(
                    f"  Dimensionality filter: axis {d} traversing component covers "
                    f"{fraction*100:.1f}% < {self.min_dim_fraction*100:.0f}%; "
                    "treated as a spurious connection and ignored."
                )

        return traverses

    # ==================================================================
    # Barrier evaluation: min-bottleneck Dijkstra on the triple stack
    # ==================================================================
    def _compute_barrier(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        pore_topo:   np.ndarray,
        axis: int,
        e_min: float,
        cell_A: np.ndarray,
    ) -> dict:
        """
        Min-bottleneck Dijkstra path search on the triple stack.

        Parameters
        ----------
        energy_grid : (nx,ny,nz)  LJ energy grid
        pore_mask   : (nx,ny,nz)  pore mask from the energy threshold filter
        pore_topo   : (nx,ny,nz)  topologically repaired mask (equals pore_mask
                                  when dilation_iter = 0)
        axis        : int         traversal axis (0=a, 1=b, 2=c)
        e_min       : float       global pore energy minimum (kJ/mol)
        cell_A      : (3,3)       cell matrix in A; rows are lattice vectors

        The original pore_mask is tried first. If no path is found and
        pore_topo differs from pore_mask, the search is repeated on the
        repaired mask with the energy of dilation-only voxels clamped to
        e_min + energy_threshold, so that the barrier stays within the
        threshold while narrow necks remain passable.
        """
        nx0, ny0, nz0 = energy_grid.shape
        n_d  = energy_grid.shape[axis]
        p1a  = (axis + 1) % 3
        p2a  = (axis + 2) % 3
        np1  = [nx0, ny0, nz0][p1a]
        np2  = [nx0, ny0, nz0][p2a]

        # Triple tiling of the energy grid
        e_stack_orig = np.concatenate([energy_grid] * 3, axis=axis)

        masks_to_try: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("original mask", pore_mask, e_stack_orig),
        ]

        # Retry on the repaired mask when dilation changed the topology
        if not np.array_equal(pore_topo, pore_mask):
            # Clamp dilation-only voxels to the threshold energy
            e_stack_topo = e_stack_orig.copy()
            p_topo_stack = np.concatenate([pore_topo] * 3, axis=axis)
            p_orig_stack = np.concatenate([pore_mask] * 3, axis=axis)
            dilated_only = p_topo_stack & ~p_orig_stack
            e_stack_topo[dilated_only] = e_min + self.energy_threshold
            masks_to_try.append(("repaired mask", pore_topo, e_stack_topo))

        for label, p_mask_1copy, e_stack in masks_to_try:
            p_stack = np.concatenate([p_mask_1copy] * 3, axis=axis)
            sn      = list(e_stack.shape)

            outcome = self._dijkstra_3copy(
                e_stack, p_stack, sn,
                axis, n_d, p1a, p2a, np1, np2,
            )

            if outcome is None:
                print(f"    [{label}] no path reaching the end region")
                continue

            max_e, path = outcome

            if len(path) < self.min_path_nodes:
                print(
                    f"    [{label}] path nodes {len(path)} < {self.min_path_nodes}; "
                    "rejected as invalid"
                )
                continue

            if label == "repaired mask":
                print("    Valid path found on the repaired mask")

            coords, energies = self._path_to_coords(
                path, axis, p1a, p2a, n_d, np1, np2, e_stack, cell_A
            )
            return {
                'barrier':           float(max_e - e_min),
                'transition_energy': float(max_e),
                'min_energy':        float(e_min),
                'path_length':       float(np.linalg.norm(cell_A[axis])),
                'n_path_nodes':      len(path),
                'path_coords':       coords,
                'path_energies':     energies,
            }

        return _empty_result(e_min)

    def _dijkstra_3copy(
        self,
        e_stack: np.ndarray,
        p_stack: np.ndarray,
        sn:      list[int],
        axis:    int,
        n_d:     int,
        p1a:     int,
        p2a:     int,
        np1:     int,
        np2:     int,
    ) -> tuple[float, list[tuple[int, int, int]]] | None:
        """
        Min-bottleneck Dijkstra search on the triple stack.

        dist[v] is the maximum energy along the path reaching v (bottleneck
        energy); the heap entry (max_e_so_far, position) makes low-bottleneck
        paths expand first.

        Traversal axis: no PBC, coordinate in [0, 3*n_d).
        Transverse axes: PBC, coordinates taken modulo np1 / np2.
        Start region: traversal coordinate in [0, bw).
        End region:   traversal coordinate in [3*n_d - bw, 3*n_d).

        Returns
        -------
        (best_max_energy, reconstructed_path) or None
        """
        bw      = max(1, int(n_d * self.boundary_fraction))
        end_min = 3 * n_d - bw

        nbrs26 = [
            (di, dj, dk)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            for dk in (-1, 0, 1)
            if (di, dj, dk) != (0, 0, 0)
        ]

        dist    = np.full(sn, np.inf)
        visited = np.zeros(sn, dtype=bool)
        parent: dict[tuple, tuple] = {}
        heap:   list               = []

        # Initialize with every pore voxel in the start region
        all_pore = np.argwhere(p_stack)             # (K, 3)
        trav_all = all_pore[:, axis]
        for v in all_pore[trav_all < bw]:
            pos = (int(v[0]), int(v[1]), int(v[2]))
            e   = float(e_stack[pos])
            if e < dist[pos]:
                dist[pos] = e
                heapq.heappush(heap, (e, pos))

        if not heap:
            return None

        best_max_e: float | None = None
        best_end:   tuple | None = None

        # Main loop
        while heap:
            cur_max_e, cur = heapq.heappop(heap)
            ix, iy, iz = cur
            if visited[ix, iy, iz]:
                continue
            visited[ix, iy, iz] = True

            # Early exit at the end region: by heap monotonicity the first
            # popped end-region node already carries the global minimum
            # bottleneck energy.
            if cur[axis] >= end_min:
                path: list[tuple] = []
                c, seen = cur, set()
                while c in parent and c not in seen:
                    path.append(c); seen.add(c); c = parent[c]
                path.append(c)
                path.reverse()
                return cur_max_e, path

            # Neighbour expansion (26-connectivity)
            for di, dj, dk in nbrs26:
                delta = (di, dj, dk)

                # Traversal axis: finite boundaries, no PBC
                new_trav = cur[axis] + delta[axis]
                if new_trav < 0 or new_trav >= 3 * n_d:
                    continue

                # Transverse axes: PBC (modulo the unit-cell voxel counts)
                new_p1 = (cur[p1a] + delta[p1a]) % np1
                new_p2 = (cur[p2a] + delta[p2a]) % np2

                nb_list        = [0, 0, 0]
                nb_list[axis]  = new_trav
                nb_list[p1a]   = new_p1
                nb_list[p2a]   = new_p2
                nb = (nb_list[0], nb_list[1], nb_list[2])

                if not p_stack[nb] or visited[nb]:
                    continue

                ne        = float(e_stack[nb])
                new_max_e = cur_max_e if ne <= cur_max_e else ne

                if new_max_e < dist[nb]:
                    dist[nb]   = new_max_e
                    parent[nb] = cur
                    heapq.heappush(heap, (new_max_e, nb))

        return None  # end region unreachable

    def _path_to_coords(
        self,
        path:    list[tuple],
        axis:    int,
        p1a:     int,
        p2a:     int,
        n_d:     int,
        np1:     int,
        np2:     int,
        e_stack: np.ndarray,
        cell_A:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a voxel-index path on the triple stack into Cartesian
        coordinates and the corresponding energy sequence.

        Traversal axis: fractional coordinate (t + 0.5)/n_d in (0, 3), already
        linear, no correction required.

        Transverse axes: the search applies PBC, so consecutive path nodes may
        differ by about np1 or np2 in index, which would appear as a straight
        line crossing the whole cell. Minimum image unwrapping removes this:
        for each step d = p1[i] - p1[i-1], a value above np1/2 subtracts np1
        from the cumulative offset and a value below -np1/2 adds np1, giving
        frac[p1a] = (p1 + offset + 0.5)/np1, which may leave [0,1] and
        represents the continuous path position in the periodic crystal.
        Path nodes are 26-connected, so regular steps change an index by at
        most 1 and are unambiguously distinguished from boundary jumps.
        """
        coords:   list[np.ndarray] = []
        energies: list[float]      = []

        p1_cum = 0   # cumulative unwrapping offset along perp1 (voxels)
        p2_cum = 0   # cumulative unwrapping offset along perp2 (voxels)

        for i, p in enumerate(path):
            t  = p[axis]
            p1 = p[p1a]
            p2 = p[p2a]

            if i > 0:
                prev = path[i - 1]
                # perp1 minimum image unwrapping
                d1 = p1 - prev[p1a]
                if   d1 >  np1 * 0.5:  p1_cum -= np1   # high-to-low PBC jump
                elif d1 < -np1 * 0.5:  p1_cum += np1   # low-to-high PBC jump
                # perp2 minimum image unwrapping
                d2 = p2 - prev[p2a]
                if   d2 >  np2 * 0.5:  p2_cum -= np2
                elif d2 < -np2 * 0.5:  p2_cum += np2

            frac       = [0.0, 0.0, 0.0]
            frac[axis] = (t            + 0.5) / n_d    # traversal axis: linear
            frac[p1a]  = (p1 + p1_cum + 0.5) / np1    # unwrapped, may leave [0,1]
            frac[p2a]  = (p2 + p2_cum + 0.5) / np2
            coords.append(np.array(frac) @ cell_A)
            energies.append(float(e_stack[p]))

        coords_arr   = np.array(coords)    # (N, 3) A
        energies_arr = np.array(energies)  # (N,)   kJ/mol

        # ==============================================================
        # Transverse periodicity correction (intersecting-channel systems)
        # ==============================================================
        #
        # In frameworks with intersecting channels (e.g. MFI) the search may
        # switch between parallel sinusoidal channels at an intersection,
        # producing a net transverse drift so that the path is no longer
        # periodic. The correction extracts the middle third of the path by
        # node index (unaffected by local backtracking along the traversal
        # axis), removes the residual drift within that segment with a linear
        # ramp, and replicates the corrected segment three times along the
        # traversal axis, restoring strict periodicity with an identical
        # energy profile per replica. The barrier value, taken from the full
        # Dijkstra path, is unaffected.
        N = len(coords_arr)
        if N >= 30:
            cl_p1 = float(np.linalg.norm(cell_A[p1a]))
            cl_p2 = float(np.linalg.norm(cell_A[p2a]))
            drift_p1 = abs(float(coords_arr[-1, p1a] - coords_arr[0, p1a]))
            drift_p2 = abs(float(coords_arr[-1, p2a] - coords_arr[0, p2a]))

            if drift_p1 > 0.20 * cl_p1 or drift_p2 > 0.20 * cl_p2:
                print(
                    f"    Periodic reconstruction: transverse drift "
                    f"p1={drift_p1:.1f} A (>{0.20*cl_p1:.1f} A)  "
                    f"p2={drift_p2:.1f} A (>{0.20*cl_p2:.1f} A)"
                )

                # Step 1: take the middle third of the path by node index
                i1 = N // 3
                i2 = (2 * N) // 3
                seg_c = coords_arr[i1:i2].copy()   # (M, 3)
                seg_e = energies_arr[i1:i2].copy()  # (M,)
                n_seg = len(seg_c)

                # Step 2: remove the transverse drift within the segment
                # with a monotonic linear ramp from 0 to 1
                linramp = np.linspace(0., 1., n_seg)
                for perp, cl in ((p1a, cl_p1), (p2a, cl_p2)):
                    d_seg = float(seg_c[-1, perp] - seg_c[0, perp])
                    if abs(d_seg) > 0.05 * cl:       # correct only above 5% drift
                        seg_c[:, perp] -= linramp * d_seg

                # Step 3: replicate three times along the traversal axis
                x_full_start = float(coords_arr[0,  axis])
                x_seg_start  = float(seg_c[0,  axis])
                x_seg_end    = float(seg_c[-1, axis])
                L_seg = max(x_seg_end - x_seg_start, 1e-6)

                all_c: list[np.ndarray] = []
                all_e: list[np.ndarray] = []
                for n in range(3):
                    rep = seg_c.copy()
                    # Shift the traversal coordinate of each replica
                    rep[:, axis] = seg_c[:, axis] - x_seg_start \
                                   + x_full_start + n * L_seg
                    all_c.append(rep)
                    all_e.append(seg_e.copy())

                coords_arr   = np.vstack(all_c)
                energies_arr = np.concatenate(all_e)
                print(f"    Periodic reconstruction completed: {n_seg} nodes x 3 replicas, "
                      f"traversal span {L_seg:.1f} A x 3 = {3*L_seg:.1f} A")

        return coords_arr, energies_arr

    # ==================================================================
    # File output
    # ==================================================================
    def _save_barrier_txt(self, barriers: dict, prefix: str) -> None:
        fpath = f"{prefix}_diffusion_barriers.txt"
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Diffusion barriers (triple-stack min-bottleneck Dijkstra)\n")
            f.write("=" * 70 + "\n\n")
            for ax in ('X', 'Y', 'Z'):
                info = barriers[ax]
                if info['barrier'] is not None:
                    f.write(
                        f"{ax} direction: barrier = {info['barrier']:.4f} kJ/mol  "
                        f"transition energy = {info['transition_energy']:.4f} kJ/mol  "
                        f"path nodes = {info['n_path_nodes']}\n"
                    )
                else:
                    f.write(f"{ax} direction: no traversing path found\n")
        print(f"  Saved: {os.path.basename(fpath)}")

    def _save_path_txts(self, barriers: dict, prefix: str) -> None:
        for ax, info in barriers.items():
            if info.get('path_coords') is None:
                continue
            fpath = f"{prefix}_path_{ax}.txt"
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(
                    f"# Minimum Energy Path along {ax}\n"
                    f"# n_nodes = {info['n_path_nodes']}\n"
                    f"# barrier = {info['barrier']:.4f} kJ/mol\n"
                    f"# X(A)        Y(A)        Z(A)        Energy(kJ/mol)\n"
                )
                for c, e in zip(info['path_coords'], info['path_energies']):
                    f.write(
                        f"{c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}  {e:12.6f}\n"
                    )
            print(f"  Saved: {os.path.basename(fpath)}")

    # ==================================================================
    # Visualization
    # ==================================================================
    def _visualize(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        barriers:    dict,
        e_min:       float,
        prefix:      str,
    ) -> None:
        """Write the unit-cell pore figure and the MEP path figure for each direction."""
        self._plot_pore_scatter(energy_grid, pore_mask, cell_A, prefix)
        for ax_idx, ax in enumerate(('X', 'Y', 'Z')):
            info = barriers[ax]
            if info.get('path_coords') is not None:
                self._plot_mep(
                    energy_grid, pore_mask, cell_A,
                    ax, ax_idx, info, prefix
                )

    def _pore_coords_1copy(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        max_pts:     int = 60_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cartesian coordinates and energies of unit-cell pore voxels (subsampled)."""
        nx, ny, nz = energy_grid.shape
        idx = np.argwhere(pore_mask)              # (N, 3)
        frac = (idx + 0.5) / np.array([nx, ny, nz])
        cart = frac @ cell_A
        ene  = energy_grid[pore_mask]

        if len(cart) > max_pts:
            sel  = np.random.choice(len(cart), max_pts, replace=False)
            cart = cart[sel]
            ene  = ene[sel]
        return cart, ene

    def _pore_coords_3copy(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        axis:        int,
        n_d:         int,
        np1:         int,
        np2:         int,
        p1a:         int,
        p2a:         int,
        max_pts:     int = 60_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pore voxel coordinates and energies on the triple stack (same frame as the MEP path)."""
        p_stack = np.concatenate([pore_mask] * 3, axis=axis)
        e_stack = np.concatenate([energy_grid] * 3, axis=axis)
        idx = np.argwhere(p_stack)                # (N, 3)

        frac       = np.zeros((len(idx), 3))
        frac[:,  axis] = (idx[:, axis] + 0.5) / n_d
        frac[:, p1a]   = (idx[:, p1a]  + 0.5) / np1
        frac[:, p2a]   = (idx[:, p2a]  + 0.5) / np2
        cart = frac @ cell_A
        ene  = e_stack[tuple(idx.T)]

        if len(cart) > max_pts:
            sel  = np.random.choice(len(cart), max_pts, replace=False)
            cart = cart[sel]
            ene  = ene[sel]
        return cart, ene

    # ==================================================================
    # Isosurface mesh (marching cubes)
    # ==================================================================
    @staticmethod
    def _pore_isosurface_mesh(
        pore_mask: np.ndarray,
        cell_A:    np.ndarray,
        sigma:     float = 1.0,
        level:     float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Extract the pore envelope isosurface from the boolean voxel mask.

        The binary mask is smoothed with a Gaussian filter (sigma controls the
        surface smoothness), the triangular isosurface is extracted at `level`
        by marching cubes, and voxel indices are converted to Cartesian
        coordinates in A.

        Returns
        -------
        (cart_verts, faces), or (None, None) if scikit-image is unavailable or
        the extraction fails.
          cart_verts : (N, 3) ndarray  vertex Cartesian coordinates
          faces      : (M, 3) ndarray  triangle vertex indices
        """
        try:
            from skimage.measure import marching_cubes
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return None, None

        nx, ny, nz = pore_mask.shape
        smooth = gaussian_filter(pore_mask.astype(np.float32), sigma=sigma)

        try:
            verts_vox, faces, _, _ = marching_cubes(
                smooth, level=level, allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            return None, None

        if len(verts_vox) == 0 or len(faces) == 0:
            return None, None

        # Voxel indices -> fractional -> Cartesian coordinates
        frac_verts = verts_vox / np.array([nx, ny, nz], dtype=float)
        cart_verts = frac_verts @ cell_A
        return cart_verts, faces.astype(np.int32)

    def _pore_isosurface_tiled(
        self,
        pore_mask:  np.ndarray,
        cell_A:     np.ndarray,
        axis_idx:   int,
        path_frac:  np.ndarray,   # unwrapped path in fractional coordinates (N, 3)
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Tile the unit-cell isosurface to cover the displayed path range
        (background of the MEP path figure): three replicas along the
        traversal axis, and along the transverse axes as many replicas as the
        fractional path range requires, each offset by the lattice vectors.
        """
        cart_base, faces_base = self._pore_isosurface_mesh(pore_mask, cell_A)
        if cart_base is None:
            return None, None

        p1a = (axis_idx + 1) % 3
        p2a = (axis_idx + 2) % 3

        p1_lo = max(int(np.floor(path_frac[:, p1a].min())) - 1, -2)
        p1_hi = min(int(np.ceil( path_frac[:, p1a].max())) + 1,  3)
        p2_lo = max(int(np.floor(path_frac[:, p2a].min())) - 1, -2)
        p2_hi = min(int(np.ceil( path_frac[:, p2a].max())) + 1,  3)

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        n_so_far = 0

        for n_t in range(3):
            for n1 in range(p1_lo, p1_hi + 1):
                for n2 in range(p2_lo, p2_hi + 1):
                    offset = (n_t * cell_A[axis_idx] +
                              n1  * cell_A[p1a] +
                              n2  * cell_A[p2a])
                    all_verts.append(cart_base + offset)
                    all_faces.append(faces_base + n_so_far)
                    n_so_far += len(cart_base)

        return np.vstack(all_verts), np.vstack(all_faces)

    def _plot_pore_scatter(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        prefix:      str,
    ) -> None:
        """Pore figure showing the transparent envelope isosurface."""
        fig = go.Figure()

        verts, faces = self._pore_isosurface_mesh(pore_mask, cell_A)
        if verts is not None:
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                opacity=0.35,
                color='steelblue',
                flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3,
                              roughness=0.5, fresnel=0.2),
                lightposition=dict(x=1000, y=1000, z=1000),
                name='Pore envelope',
                showlegend=False,
            ))
        else:
            # Fallback to a sparse scatter when scikit-image is unavailable
            cart, ene = self._pore_coords_1copy(energy_grid, pore_mask, cell_A,
                                                max_pts=15_000)
            fig.add_trace(go.Scatter3d(
                x=cart[:, 0], y=cart[:, 1], z=cart[:, 2],
                mode='markers',
                marker=dict(size=1.5, color='steelblue', opacity=0.25),
                name='Pore (fallback)',
            ))

        fig.update_layout(
            title="Accessible Pore Channels - transparent envelope",
            scene=dict(
                xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)",
                aspectmode='data',
                bgcolor='white',
                xaxis=dict(backgroundcolor='white', gridcolor='#e0e0e0'),
                yaxis=dict(backgroundcolor='white', gridcolor='#e0e0e0'),
                zaxis=dict(backgroundcolor='white', gridcolor='#e0e0e0'),
            ),
            paper_bgcolor='white',
            width=1300, height=950,
        )
        fpath = f"{prefix}_pore_scatter.html"
        plot(fig, filename=fpath, auto_open=False)
        print(f"  Saved: {os.path.basename(fpath)}")

    def _pore_coords_tiled(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        axis_idx:    int,
        path_frac:   np.ndarray,   # unwrapped path in fractional coordinates, (N, 3)
        max_pts:     int = 80_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate pore background tiles matching the coordinate range of the
        unwrapped path.

        Traversal axis: three replicas covering the full triple stack.
        Transverse axes: integer offsets from the floor/ceil range of the path
        fractional coordinates, with one extra replica on each side.
        Point count is capped at max_pts to keep rendering responsive.
        """
        nx0, ny0, nz0 = energy_grid.shape
        n_d  = [nx0, ny0, nz0][axis_idx]
        p1a  = (axis_idx + 1) % 3
        p2a  = (axis_idx + 2) % 3
        np1  = [nx0, ny0, nz0][p1a]
        np2  = [nx0, ny0, nz0][p2a]

        # Transverse tile range from the unwrapped path (one replica margin)
        p1_lo = int(np.floor(path_frac[:, p1a].min())) - 1
        p1_hi = int(np.ceil( path_frac[:, p1a].max())) + 1
        p2_lo = int(np.floor(path_frac[:, p2a].min())) - 1
        p2_hi = int(np.ceil( path_frac[:, p2a].max())) + 1

        # Cap the number of tiles per direction
        p1_lo = max(p1_lo, -3);  p1_hi = min(p1_hi, 4)
        p2_lo = max(p2_lo, -3);  p2_hi = min(p2_hi, 4)

        idx      = np.argwhere(pore_mask)        # (K, 3)
        ene_orig = energy_grid[pore_mask]        # (K,)
        all_carts: list[np.ndarray] = []
        all_enes:  list[np.ndarray] = []

        for n_t in range(3):
            for n1 in range(p1_lo, p1_hi + 1):
                for n2 in range(p2_lo, p2_hi + 1):
                    frac = np.zeros((len(idx), 3))
                    frac[:, axis_idx] = (idx[:, axis_idx] + 0.5) / n_d + n_t
                    frac[:, p1a]      = (idx[:, p1a]      + 0.5) / np1 + n1
                    frac[:, p2a]      = (idx[:, p2a]      + 0.5) / np2 + n2
                    all_carts.append(frac @ cell_A)
                    all_enes.append(ene_orig)

        cart_all = np.vstack(all_carts)
        ene_all  = np.concatenate(all_enes)

        if len(cart_all) > max_pts:
            sel      = np.random.choice(len(cart_all), max_pts, replace=False)
            cart_all = cart_all[sel]
            ene_all  = ene_all[sel]

        return cart_all, ene_all

    def _plot_mep(
        self,
        energy_grid: np.ndarray,
        pore_mask:   np.ndarray,
        cell_A:      np.ndarray,
        ax:          str,
        axis_idx:    int,
        info:        dict,
        prefix:      str,
    ) -> None:
        """
        MEP path figure.

        Path coordinates are already unwrapped by _path_to_coords, so no PBC
        jumps remain along the transverse axes; the pore background is tiled
        to match the displayed path range.
        """
        c = info['path_coords']
        e = info['path_energies']

        cell_A_inv = np.linalg.inv(cell_A)
        path_frac  = c @ cell_A_inv

        ax_color = {'X': 'crimson', 'Y': 'forestgreen', 'Z': 'royalblue'}[ax]

        fig = go.Figure()

        # Tiled transparent envelope isosurface
        iso_v, iso_f = self._pore_isosurface_tiled(
            pore_mask, cell_A, axis_idx, path_frac
        )
        if iso_v is not None:
            fig.add_trace(go.Mesh3d(
                x=iso_v[:, 0], y=iso_v[:, 1], z=iso_v[:, 2],
                i=iso_f[:, 0], j=iso_f[:, 1], k=iso_f[:, 2],
                opacity=0.18,
                color='steelblue',
                flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3,
                              roughness=0.5, fresnel=0.2),
                lightposition=dict(x=1000, y=1000, z=1000),
                name='Pore envelope',
                showlegend=True,
            ))
        else:
            # Fallback to a sparse scatter when scikit-image is unavailable
            bg_cart, bg_ene = self._pore_coords_tiled(
                energy_grid, pore_mask, cell_A, axis_idx, path_frac,
                max_pts=15_000,
            )
            fig.add_trace(go.Scatter3d(
                x=bg_cart[:, 0], y=bg_cart[:, 1], z=bg_cart[:, 2],
                mode='markers',
                marker=dict(size=1.0, color='steelblue', opacity=0.15),
                name='Pore (fallback)',
            ))

        # MEP path, coloured by energy
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode='lines+markers',
            line=dict(
                color=e, colorscale='thermal', width=8,
                showscale=True,
                cmin=float(e.min()), cmax=float(e.max()),
                colorbar=dict(title="Energy<br>(kJ/mol)", x=1.12),
            ),
            marker=dict(size=3, color=e, colorscale='thermal',
                        cmin=float(e.min()), cmax=float(e.max()),
                        showscale=False),
            name=f'{ax} MEP',
        ))

        # Start and end markers
        fig.add_trace(go.Scatter3d(
            x=[c[0, 0], c[-1, 0]],
            y=[c[0, 1], c[-1, 1]],
            z=[c[0, 2], c[-1, 2]],
            mode='markers',
            marker=dict(size=10, color=[ax_color, ax_color],
                        symbol=['diamond', 'square'],
                        line=dict(color='black', width=1.5)),
            name='Start / End',
        ))

        fig.update_layout(
            title=(
                f"{ax}-direction Minimum Energy Path  "
                f"Barrier = {info['barrier']:.3f} kJ/mol"
            ),
            scene=dict(
                xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)",
                aspectmode='data',
                bgcolor='white',
            ),
            paper_bgcolor='white',
            width=1400, height=1000,
        )
        fpath = f"{prefix}_diffusion_path_{ax}.html"
        plot(fig, filename=fpath, auto_open=False)
        print(f"  Saved: {os.path.basename(fpath)}")


# ======================================================================
# Helpers
# ======================================================================
def _empty_result(e_min: float) -> dict:
    """Empty barrier result, used for non-traversable directions."""
    return {
        'barrier':           None,
        'transition_energy': None,
        'min_energy':        float(e_min),
        'path_length':       None,
        'n_path_nodes':      0,
        'path_coords':       None,
        'path_energies':     None,
    }