"""Lennard-Jones potential energy grid evaluated on a supercell and trimmed to the unit cell."""
import os
import time
import numpy as np
from ase.io import read, write

# Optional numba acceleration
USE_NUMBA = False
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    pass

if USE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _lj_numba(frac_pts, frac_atoms, sigma_atoms, eps_atoms,
                  sigma_probe, eps_probe, cell, cutoff):
        n_pts, n_at = frac_pts.shape[0], frac_atoms.shape[0]
        out = np.zeros(n_pts)
        for ip in prange(n_pts):
            fx, fy, fz = frac_pts[ip]
            total = 0.0
            for ia in range(n_at):
                dfx = frac_atoms[ia, 0] - fx
                dfy = frac_atoms[ia, 1] - fy
                dfz = frac_atoms[ia, 2] - fz
                dfx -= round(dfx); dfy -= round(dfy); dfz -= round(dfz)
                drx = dfx*cell[0,0] + dfy*cell[1,0] + dfz*cell[2,0]
                dry = dfx*cell[0,1] + dfy*cell[1,1] + dfz*cell[2,1]
                drz = dfx*cell[0,2] + dfy*cell[1,2] + dfz*cell[2,2]
                r = (drx*drx + dry*dry + drz*drz)**0.5
                if r < 0.01:
                    total = 1000.0
                    break
                if r <= cutoff:
                    sm = 0.5 * (sigma_atoms[ia] + sigma_probe)
                    em = (eps_atoms[ia] * eps_probe)**0.5
                    sr6 = (sm / r) ** 6
                    total += 4.0 * em * sr6 * (sr6 - 1.0)
            out[ip] = min(total, 1000.0)
        return out
else:
    _lj_numba = None


class GridGenerator:
    """Three-dimensional LJ potential energy grid (supercell evaluation, unit-cell trimming)."""

    # sigma (nm), epsilon (kJ/mol); framework entries are also used for the
    # host atoms in the Lorentz-Berthelot mixing rules.
    FF_PARAMS = {
        "o":   (0.3304, 0.442329411),
        "si":  (0.2310, 0.18458107),
        "ch4": (0.3730, 1.230540467),
        "xe":  (0.4100, 1.837496239),
    }

    def __init__(self, grid_spacing=0.2, cutoff=14.0, probe_type="ch4",
                 min_supercell_length=28.0):
        self.grid_spacing = grid_spacing
        self.cutoff = cutoff
        self.probe_type = probe_type.lower()
        self.min_supercell_length = min_supercell_length

    def build(self, cif_file: str, output_prefix: str,
              save_vts: bool = True, save_grid_txt: bool = True) -> dict:
        """
        Supercell expansion, grid evaluation, unit-cell trimming, file output.

        Parameters
        ----------
        cif_file      : path to the input CIF structure
        output_prefix : prefix for all output file paths
        save_vts      : write the ParaView structured grid file (*.vts)
        save_grid_txt : write the energy grid text file (*_energy_grid.txt)

        Returns
        -------
        dict with keys:
          'supercell'      : supercell grid data (grid_energies, cell, frac_x/y/z, n_grid)
          'unit_cell'      : unit-cell grid data (as above, plus grid_file)
          'scale_factors'  : supercell replication factors [nx, ny, nz]
          'original_cell'  : original unit cell matrix (3x3, nm)
          'output_prefix'  : output path prefix
        """
        print("\nStep 1: supercell expansion")
        atoms = read(cif_file)
        original_cell = atoms.get_cell() / 10.0          # Angstrom -> nm
        lengths = atoms.get_cell_lengths_and_angles()[:3]
        print(f"Unit cell: a={lengths[0]:.2f}, b={lengths[1]:.2f}, c={lengths[2]:.2f} A")

        scale = [max(1, int(np.ceil(self.min_supercell_length / L))) for L in lengths]
        print(f"Supercell factors: {scale}")

        supercell = atoms * scale
        new_L = supercell.get_cell_lengths_and_angles()[:3]
        print(f"Supercell: a={new_L[0]:.2f}, b={new_L[1]:.2f}, c={new_L[2]:.2f} A")

        print("\nStep 2: potential energy grid (supercell)")
        super_data = self._compute_grid(supercell)

        print("\nStep 2.5: trimming to unit cell")
        unit_data = self._trim_to_unit_cell(super_data, original_cell, scale)

        # Unit-cell energy grid text file (*_energy_grid.txt)
        unit_file = f"{output_prefix}_energy_grid.txt"
        if save_grid_txt:
            self._save_grid(unit_data, unit_file, "Unit Cell")
        else:
            print("  Skipped: energy grid text file (output.save_grid_txt = false)")
        unit_data['grid_file'] = unit_file   # Path always recorded for downstream cleanup

        # Export the unit-cell grid as VTS for ParaView inspection
        if save_vts:
            self.save_vts(unit_data, f"{output_prefix}_energy_grid.vts")
        else:
            print("  Skipped: VTS file (output.save_vts = false)")

        return {
            'supercell': super_data,
            'unit_cell': unit_data,
            'scale_factors': scale,
            'original_cell': original_cell,
            'output_prefix': output_prefix,
        }

    @staticmethod
    def save_vts(grid_data: dict, output_file: str, energy_cap: float = 1000.0):
        """
        Export grid data as a VTK XML StructuredGrid (.vts) file, valid for
        arbitrary (including non-orthogonal) cells.

        Parameters
        ----------
        grid_data   : the 'unit_cell' sub-dictionary returned by GridGenerator.build()
        output_file : output path, e.g. "xxx_energy_grid.vts"
        energy_cap  : display cut-off for high-energy points (repulsive/core regions), kJ/mol
        """
        cell_nm = grid_data['cell']          # (3,3) nm
        cell_A  = cell_nm * 10.0             # nm -> Angstrom
        fx = grid_data['frac_x']
        fy = grid_data['frac_y']
        fz = grid_data['frac_z']
        g  = grid_data['grid_energies']
        n1, n2, n3 = grid_data['n_grid']

        # VTK StructuredGrid ordering: i(x) varies fastest, k(z) slowest,
        # so g[i,j,k] is unrolled in k,j,i order; r = [fi,fj,fk] @ cell_A
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')   # shape (n1,n2,n3)
        frac_pts = np.stack([FX, FY, FZ], axis=-1)             # (n1,n2,n3,3)
        cart_pts = frac_pts @ cell_A                            # (n1,n2,n3,3) A

        # Transpose to (n3,n2,n1,3) before ravel to match the VTK point order
        pts_vtk = cart_pts.transpose(2, 1, 0, 3).reshape(-1, 3)   # (N,3)
        e_vtk   = np.clip(g.transpose(2, 1, 0).ravel(), -1e6, energy_cap)

        N = n1 * n2 * n3
        ext = f"0 {n1-1} 0 {n2-1} 0 {n3-1}"

        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">',
            f'  <StructuredGrid WholeExtent="{ext}">',
            f'    <Piece Extent="{ext}">',
            '      <Points>',
            '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        ]

        # Coordinate block: one point (x y z) per line
        coord_lines = '\n'.join(
            f'          {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for p in pts_vtk
        )
        lines.append(coord_lines)

        lines += [
            '        </DataArray>',
            '      </Points>',
            '      <PointData Scalars="Energy_kJmol">',
            '        <DataArray type="Float32" Name="Energy_kJmol" format="ascii">',
            '          ' + ' '.join(f'{e:.4f}' for e in e_vtk),
            '        </DataArray>',
            '      </PointData>',
            '    </Piece>',
            '  </StructuredGrid>',
            '</VTKFile>',
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        valid = e_vtk < energy_cap * 0.99
        print(f"VTS file saved: {output_file}")
        print(f"  Grid size: {n1}x{n2}x{n3} = {N:,} points")
        if valid.any():
            print(f"  Energy range (accessible region): [{e_vtk[valid].min():.1f}, {e_vtk[valid].max():.1f}] kJ/mol")

    # ------------------------------------------------------------------
    def _compute_grid(self, atoms) -> dict:
        """Evaluate the LJ potential energy grid for a given atomic system."""
        cell = atoms.get_cell() / 10.0           # nm
        positions = atoms.get_positions() / 10.0 # nm
        symbols = atoms.get_chemical_symbols()

        cell_inv = np.linalg.inv(cell)
        sigma_p, eps_p = self.FF_PARAMS[self.probe_type]
        n_at = len(symbols)
        sigma_a, eps_a = np.zeros(n_at), np.zeros(n_at)
        frac_a = np.zeros((n_at, 3))

        for i, (pos, sym) in enumerate(zip(positions, symbols)):
            key = sym.lower()[:2] if len(sym) > 1 else sym.lower()[0]
            if key not in self.FF_PARAMS:
                key = "o"
            sigma_a[i], eps_a[i] = self.FF_PARAMS[key]
            frac_a[i] = pos @ cell_inv

        # Grid dimensions
        sp_nm = self.grid_spacing / 10.0
        n_grid = [max(1, int(np.ceil(np.linalg.norm(cell[i]) / sp_nm))) for i in range(3)]
        n1, n2, n3 = n_grid
        print(f"Grid size: {n1}x{n2}x{n3} = {n1*n2*n3:,} points")

        fx = np.linspace(0.0, 1.0, n1, endpoint=False)
        fy = np.linspace(0.0, 1.0, n2, endpoint=False)
        fz = np.linspace(0.0, 1.0, n3, endpoint=False)

        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        frac_pts = np.vstack([FX.ravel(), FY.ravel(), FZ.ravel()]).T

        t0 = time.time()
        if USE_NUMBA:
            energies = _lj_numba(frac_pts, frac_a, sigma_a, eps_a,
                                  sigma_p, eps_p, cell, self.cutoff / 10.0)
            print("Numba parallel evaluation enabled")
        else:
            energies = self._lj_numpy(frac_pts, frac_a, sigma_a, eps_a,
                                      sigma_p, eps_p, cell, self.cutoff / 10.0)
        print(f"Computation time: {time.time()-t0:.1f} s")

        grid_e = energies.reshape(n1, n2, n3)
        valid = grid_e < 1000
        if valid.any():
            print(f"Energy range: [{grid_e[valid].min():.1f}, {grid_e[valid].max():.1f}] kJ/mol")

        return {'grid_energies': grid_e, 'cell': cell,
                'frac_x': fx, 'frac_y': fy, 'frac_z': fz, 'n_grid': n_grid}

    @staticmethod
    def _lj_numpy(frac_pts, frac_a, sigma_a, eps_a, sigma_p, eps_p, cell, cutoff_nm):
        n_pts = len(frac_pts)
        out = np.zeros(n_pts)
        for i in range(n_pts):
            if i % 20000 == 0 and i > 0:
                print(f"  Progress: {i}/{n_pts}")
            df = frac_a - frac_pts[i]
            df -= np.round(df)
            dr = df @ cell
            r = np.linalg.norm(dr, axis=1)
            if np.any(r < 0.01):
                out[i] = 1000.0
                continue
            mask = r <= cutoff_nm
            if mask.any():
                sm = 0.5 * (sigma_a[mask] + sigma_p)
                em = np.sqrt(eps_a[mask] * eps_p)
                sr6 = (sm / r[mask]) ** 6
                out[i] = min(np.sum(4.0 * em * sr6 * (sr6 - 1.0)), 1000.0)
        return out

    @staticmethod
    def _trim_to_unit_cell(super_data: dict, original_cell, scale) -> dict:
        """Trim the supercell energy grid to the original unit cell."""
        g = super_data['grid_energies']
        n1, n2, n3 = super_data['n_grid']
        nu = [n1//scale[0], n2//scale[1], n3//scale[2]]
        print(f"Supercell grid {[n1,n2,n3]} -> unit cell grid {nu}")

        unit_e = g[:nu[0], :nu[1], :nu[2]].copy()
        valid = unit_e < 1000
        if valid.any():
            print(f"Unit cell energy range: [{unit_e[valid].min():.1f}, {unit_e[valid].max():.1f}] kJ/mol")

        return {
            'grid_energies': unit_e,
            'cell': original_cell,
            'frac_x': np.linspace(0.0, 1.0, nu[0], endpoint=False),
            'frac_y': np.linspace(0.0, 1.0, nu[1], endpoint=False),
            'frac_z': np.linspace(0.0, 1.0, nu[2], endpoint=False),
            'n_grid': nu,
        }

    @staticmethod
    def _save_grid(grid_data: dict, output_file: str, label: str = ""):
        """Write grid data to a text file (vectorized)."""
        cell = grid_data['cell']
        fx, fy, fz = grid_data['frac_x'], grid_data['frac_y'], grid_data['frac_z']
        g = grid_data['grid_energies']
        n1, n2, n3 = grid_data['n_grid']

        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        frac_pts = np.vstack([FX.ravel(), FY.ravel(), FZ.ravel()]).T
        cart_pts = frac_pts @ (cell * 10.0)           # nm -> Angstrom
        data = np.column_stack([cart_pts, g.ravel()])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Energy Grid ({label})\n")
            f.write(f"# Grid size: {n1} x {n2} x {n3}\n")
            for i, v in enumerate(['a', 'b', 'c']):
                f.write(f"#   {v} = [{cell[i,0]:.6f}, {cell[i,1]:.6f}, {cell[i,2]:.6f}] nm\n")
            f.write("# X(A)         Y(A)         Z(A)         Energy(kJ/mol)\n")
            np.savetxt(f, data, fmt='%.6f')

        print(f"Saved: {output_file}")
