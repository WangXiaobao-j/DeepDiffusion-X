"""Descriptor nomenclature: internal keys and their display symbols, units, and definitions."""
from collections import OrderedDict

# -- Descriptor families (used for grouping in the glossary panel) ----------
FAMILY_GEOMETRY = "Pore geometry"
FAMILY_TOPOLOGY = "Pathway topology"
FAMILY_ENERGETICS = "Energy landscape"
FAMILY_BARRIER = "Barrier structure"

FAMILY_ORDER = [FAMILY_GEOMETRY, FAMILY_TOPOLOGY, FAMILY_ENERGETICS, FAMILY_BARRIER]


class Descriptor:
    """Nomenclature record for one descriptor. Immutable by convention."""

    __slots__ = ("key", "symbol", "mathtext", "ascii", "name", "unit", "family", "definition")

    def __init__(self, key, symbol, mathtext, ascii_, name, unit, family, definition):
        self.key = key                 # internal column name - never rename
        self.symbol = symbol           # Unicode symbol shown in the GUI
        self.mathtext = mathtext       # matplotlib mathtext for figures
        self.ascii = ascii_            # 7-bit fallback
        self.name = name               # full descriptor name
        self.unit = unit               # "" for dimensionless quantities
        self.family = family
        self.definition = definition   # one-sentence operational definition

    # -- Convenience renderings --------------------------------------------
    @property
    def html(self) -> str:
        """
        Rich-text rendering for Qt widgets that accept HTML (labels, tooltips,
        chat bubbles). Descriptor symbols are set in italic, as variable names
        are in a manuscript - no subscript markup, because the symbols are flat
        strings (Ebarrier, Vacc, ASA) rather than base-plus-subscript pairs.
        """
        return f"<i>{self.symbol}</i>"

    @property
    def html_with_unit(self) -> str:
        return f"{self.html} ({self.unit})" if self.unit else self.html

    @property
    def symbol_with_unit(self) -> str:
        return f"{self.symbol} ({self.unit})" if self.unit else self.symbol

    @property
    def mathtext_with_unit(self) -> str:
        return f"{self.mathtext} ({self.unit})" if self.unit else self.mathtext

    def tooltip(self) -> str:
        """Rich-text tooltip: symbol, full name, unit, definition, internal key."""
        unit = self.unit or "dimensionless"
        return (f"<b>{self.html}</b> \u2014 {self.name}<br>"
                f"<i>Unit:</i> {unit}<br>"
                f"<i>Family:</i> {self.family}<br>"
                f"{self.definition}<br>"
                f"<span style='color:#8A8F98'>internal key: {self.key}</span>")

    def __repr__(self):
        return f"<Descriptor {self.key} -> {self.symbol}>"


# ============================================================================
# The registry. Order follows FEATURE_ORDER in feature_merge.py so that any
# iteration over DESCRIPTORS reproduces the model's feature order.
# ============================================================================
_D = [
    Descriptor(
        "barrier_official", "Ebarrier", "Ebarrier", "Ebarrier",
        "Minimax (min-bottleneck) diffusion barrier", "kJ mol\u207b\u00b9", FAMILY_BARRIER,
        "Height of the rate-limiting bottleneck along the percolating pathway, from the "
        "min-bottleneck Dijkstra search on the LJ energy grid over the 3x3x3 supercell. "
        "Distinct from the per-path E_a column of the descriptor workbook, which is the "
        "max-min range of a single unit-cell pathway.",
    ),
    Descriptor(
        "sigma_E", "\u03c3E", "\u03c3E", "sigmaE",
        "Energy-landscape roughness", "kJ mol\u207b\u00b9", FAMILY_ENERGETICS,
        "Arc-length-weighted standard deviation of the potential energy sampled along the "
        "pathway; the Zwanzig energy-disorder term.",
    ),
    Descriptor(
        "eta_main", "\u03b7main", "\u03b7main", "etamain",
        "Main-axis transport efficiency", "", FAMILY_TOPOLOGY,
        "Projection of the end-to-end displacement onto the crystallographic axis of interest, "
        "divided by the traversed pathway length; unity for a perfectly axial channel.",
    ),
    Descriptor(
        "tortuosity", "P\u03c4", "P\u03c4", "Ptau",
        "Pathway tortuosity", "", FAMILY_TOPOLOGY,
        "Ratio of the contour length of the minimum-energy pathway to the straight-line "
        "distance between its termini; \u2265 1 by construction.",
    ),
    Descriptor(
        "E_max", "Emax", "Emax", "Emax",
        "Maximum pathway energy", "kJ mol\u207b\u00b9", FAMILY_ENERGETICS,
        "Highest potential energy encountered along the pathway \u2014 the most repulsive "
        "configuration the diffusing probe must traverse.",
    ),
    Descriptor(
        "E_min_official", "Emin", "Emin", "Emin",
        "Adsorption-site baseline energy", "kJ mol\u207b\u00b9", FAMILY_ENERGETICS,
        "Reference minimum of the energy landscape recovered from the min-bottleneck "
        "pathway (transition energy minus barrier); sets a common energy zero so frameworks "
        "remain mutually comparable.",
    ),
    Descriptor(
        "peak_slope_mean", "GE,mean", "GE,mean", "GEmean",
        "Mean barrier steepness", "kJ mol\u207b\u00b9 \u212b\u207b\u00b9", FAMILY_BARRIER,
        "Mean of the steeper valley-to-peak energy gradient on either flank of each resolved "
        "barrier, evaluated over the 3D Euclidean node separation; 0 when no peak is resolved.",
    ),
    Descriptor(
        "H_entropy", "HE", "HE", "HE",
        "Differential energy entropy", "", FAMILY_ENERGETICS,
        "Shannon differential entropy of the pathway energy distribution (kJ mol\u207b\u00b9 basis, "
        "reported dimensionless); quantifies how broadly the energy states are sampled.",
    ),
    Descriptor(
        "barrier_freq", "\u03c1barrier", "\u03c1barrier", "rhobarrier",
        "Barrier number density", "\u212b\u207b\u00b9", FAMILY_BARRIER,
        "Number of prominence-resolved barriers per unit pathway length (cell-boundary "
        "artefacts removed); the hop density of the multi-jump diffusion picture.",
    ),
    Descriptor(
        "PLD", "PLD", "PLD", "PLD",
        "Pore limiting diameter", "\u212b", FAMILY_GEOMETRY,
        "Direction-resolved percolation-limited free-sphere diameter, obtained by bisection "
        "search on the voxelised free-radius grid.",
    ),
    Descriptor(
        "FDSi", "FDSi", "FDSi", "FDSi",
        "Framework density", "Si nm\u207b\u00b3", FAMILY_GEOMETRY,
        "Number of tetrahedral Si centres per unit volume, equivalently the IZA framework "
        "density in T atoms per 1000 \u212b\u00b3; a compactness measure of the silicate skeleton.",
    ),
    Descriptor(
        "Vacc", "Vacc", "Vacc", "Vacc",
        "Accessible pore volume fraction", "%", FAMILY_GEOMETRY,
        "Probe-accessible free volume as a percentage of the cell volume \u2014 the dwell "
        "space available to the guest molecule.",
    ),
    Descriptor(
        "ASA", "ASA", "ASA", "ASA",
        "Accessible surface area", "m\u00b2 g\u207b\u00b9", FAMILY_GEOMETRY,
        "Probe-accessible internal surface area; proportional to the guest\u2013wall collision "
        "frequency and hence to local trapping.",
    ),
]

DESCRIPTORS = OrderedDict((d.key, d) for d in _D)

# -- Flat lookup tables (built once) ----------------------------------------
SYMBOLS = {k: d.symbol for k, d in DESCRIPTORS.items()}
MATHTEXT = {k: d.mathtext for k, d in DESCRIPTORS.items()}
ASCII_SYMBOLS = {k: d.ascii for k, d in DESCRIPTORS.items()}
UNITS = {k: d.unit for k, d in DESCRIPTORS.items()}

# Reverse map: any known rendering of a descriptor back to its internal key.
# Legacy spellings are kept so text written before the symbols were flattened
# (underscore forms, the older *_official wording) still resolves.
_LEGACY_ALIASES = {
    "barrier_official": ["E_a", "E_a,mb", "E_a_mb", "Ea"],
    "E_min_official": ["E_min", "E_min,mb", "E_min_mb"],
    "sigma_E": ["sigma_E", "\u03c3_E"],
    "eta_main": ["eta_main", "\u03b7_main"],
    "tortuosity": ["P_tau", "P_\u03c4"],
    "E_max": ["E_max"],
    "peak_slope_mean": ["G_E,mean", "G_E_mean", "SPmean", "SP_mean"],
    "H_entropy": ["H_E"],
    "barrier_freq": ["rho_barrier", "\u03c1_barrier", "fbarrier", "f_barrier"],
    "FDSi": ["FD_Si"],
    "PLD": ["PLD1"],
    "Vacc": ["V_acc"],
    "ASA": ["A_acc"],
}

_ALIASES = {}
for _k, _d in DESCRIPTORS.items():
    for _alias in (_k, _d.symbol, _d.ascii, _d.name):
        _ALIASES[_alias.lower()] = _k
for _k, _legacy in _LEGACY_ALIASES.items():
    for _alias in _legacy:
        _ALIASES.setdefault(_alias.lower(), _k)


# ============================================================================
# Public helpers
# ============================================================================
def resolve(name: str) -> str | None:
    """Map any known rendering of a descriptor back to its internal column key."""
    return _ALIASES.get(str(name).strip().lower())


def symbol(key: str, with_unit: bool = False) -> str:
    """Unicode symbol for an internal key; unknown keys are returned unchanged."""
    d = DESCRIPTORS.get(key)
    if d is None:
        return str(key)
    return d.symbol_with_unit if with_unit else d.symbol


def mathtext(key: str, with_unit: bool = False) -> str:
    """matplotlib mathtext label for an internal key."""
    d = DESCRIPTORS.get(key)
    if d is None:
        return str(key)
    return d.mathtext_with_unit if with_unit else d.mathtext


def html(key: str, with_unit: bool = False) -> str:
    """Rich-text (subscripted) symbol for an internal key."""
    d = DESCRIPTORS.get(key)
    if d is None:
        return str(key)
    return d.html_with_unit if with_unit else d.html


def ascii_symbol(key: str) -> str:
    """7-bit-safe symbol, for file names and log lines."""
    d = DESCRIPTORS.get(key)
    return d.ascii if d else str(key)


def tooltip(key: str) -> str:
    """Rich-text tooltip for an internal key, or an empty string if unknown."""
    d = DESCRIPTORS.get(key)
    return d.tooltip() if d else ""


def label_list(keys, with_unit: bool = True, kind: str = "symbol") -> list:
    """Render a sequence of internal keys as display labels (order preserved)."""
    fn = {"symbol": symbol, "mathtext": mathtext, "html": html}.get(kind, symbol)
    return [fn(k, with_unit) for k in keys]


def rename_dataframe_columns(df, with_unit: bool = True, inplace: bool = False):
    """
    Return a copy of ``df`` whose descriptor columns carry display symbols.
    Identifier columns (Zeolite, Framework, Direction ...) are left untouched.
    Purely cosmetic - never call this on a frame headed for the model.
    """
    mapping = {k: symbol(k, with_unit) for k in DESCRIPTORS if k in df.columns}
    if not mapping:
        return df if inplace else df.copy()
    return df.rename(columns=mapping, inplace=inplace) or df


def glossary_rows() -> list:
    """
    Glossary table rows grouped by descriptor family, for the reference panel:
    ``[(family, [Descriptor, ...]), ...]``.
    """
    grouped = []
    for fam in FAMILY_ORDER:
        members = [d for d in DESCRIPTORS.values() if d.family == fam]
        if members:
            grouped.append((fam, members))
    return grouped


def nomenclature_block(keys=None) -> str:
    """
    Plain-text nomenclature block, e.g. for the LLM grounding context or a
    log header: one ``symbol = name (unit) [internal key]`` line per descriptor.
    """
    keys = keys or list(DESCRIPTORS)
    lines = []
    for k in keys:
        d = DESCRIPTORS.get(k)
        if d is None:
            continue
        unit = f" [{d.unit}]" if d.unit else ""
        lines.append(f"  {d.symbol} = {d.name}{unit}  (internal key: {d.key})")
    return "\n".join(lines)
