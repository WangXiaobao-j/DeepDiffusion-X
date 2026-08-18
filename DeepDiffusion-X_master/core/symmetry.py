"""CIF symmetry removal by P1 expansion with the unit cell strictly preserved."""
import os
import warnings
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")


class SymmetryError(Exception):
    pass


def manual_expand_symmetry_keep_cell(structure: Structure, symprec: float = 0.1) -> Structure:
    """Manually expand symmetry operations, strictly preserving the original unit cell (matches the reference algorithm)."""
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sym_ops = sga.get_space_group_operations()

    expanded_sites = []
    for site in structure:
        for sym_op in sym_ops:
            new_coords = sym_op.operate(site.frac_coords)
            new_coords = new_coords % 1.0

            is_duplicate = False
            for existing_site in expanded_sites:
                if (existing_site['species'] == site.species_string and
                        all(abs(new_coords - existing_site['coords']) < 1e-3)):
                    is_duplicate = True
                    break

            if not is_duplicate:
                expanded_sites.append({
                    'species': site.species_string,
                    'coords': new_coords
                })

    expanded_structure = Structure(
        lattice=structure.lattice,
        species=[s['species'] for s in expanded_sites],
        coords=[s['coords'] for s in expanded_sites]
    )
    return expanded_structure


def remove_symmetry(cif_path: str, output_dir: str, symprec: float = 0.1,
                     log=print) -> dict:
    """
    Remove symmetry from a single CIF file and save it as a P1 structure.

    Returns
    -------
    dict: {
        'ok': bool, 'output_path': str | None, 'space_group': str | None,
        'original_atoms': int, 'final_atoms': int, 'volume_change': float,
        'error': str
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(cif_path)
    rec = dict(ok=False, output_path=None, space_group=None,
               original_atoms=0, final_atoms=0, volume_change=0.0, error="")
    try:
        structure = Structure.from_file(cif_path)
        original_volume = structure.lattice.volume
        original_atoms = len(structure)

        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        space_group = sga.get_space_group_symbol()

        expanded_structure = manual_expand_symmetry_keep_cell(structure, symprec)

        final_volume = expanded_structure.lattice.volume
        final_atoms = len(expanded_structure)
        volume_change = abs(final_volume - original_volume) / max(original_volume, 1e-9)

        if volume_change > 1e-6:
            log(f"  Warning: {name} unit cell volume changed slightly ({volume_change:.2e})")

        output_path = os.path.join(output_dir, name)
        expanded_structure.to(filename=output_path)

        rec.update(ok=True, output_path=output_path, space_group=space_group,
                   original_atoms=original_atoms, final_atoms=final_atoms,
                   volume_change=volume_change)
        log(f"  {name}  space group={space_group}  atoms {original_atoms}->{final_atoms}")

    except Exception as exc:
        rec["error"] = str(exc)
        log(f"  {name}: symmetry removal failed: {exc}")

    return rec


def batch_remove_symmetry(input_folder: str, output_folder: str,
                           symprec: float = 0.1, log=print) -> list:
    """Batch-process all .cif files in a folder; the output folder is kept separate from the input folder."""
    if os.path.abspath(output_folder) == os.path.abspath(input_folder):
        output_folder = input_folder.rstrip("/\\") + "_no_symmetry"
        log(f"Warning: output folder equals input folder; changed to: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    results = []
    cif_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".cif"))
    for fname in cif_files:
        rec = remove_symmetry(os.path.join(input_folder, fname), output_folder,
                               symprec=symprec, log=log)
        rec["file_name"] = fname
        results.append(rec)

    n_ok = sum(r["ok"] for r in results)
    log(f"Symmetry removal complete: {n_ok}/{len(results)} succeeded. Output directory: {output_folder}")
    return results
