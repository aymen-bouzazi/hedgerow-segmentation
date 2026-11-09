"""The eleven assignment methods, as recipes.

Each method is the same small set of building blocks with different options:

* which ALS scores: stem paths from basal seeds (``basal``), stem paths from
  basal + canopy-peak seeds (``peaks``), or none (comparators);
* which field heights (none, lower-stratum, 2 m classes, requested only);
* the allowance above field height (0.5 or 1.0 m);
* whether trees without a basal seed are retried with the wider window.

The short code (left) is the name used in files and tables; the
descriptive name (``name``) is the one used in the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    """Recipe of one assignment method (see module docstring)."""

    code: str
    name: str
    graph: str | None  # "basal", "peaks" or None
    heights: str  # "none", "lower", "classes", "requested", "-"
    allowance: float | None
    wide_window: bool
    summary: str


METHODS = [
    Method(
        "Nearest_stem_XY",
        "Nearest-stem proximity",
        None,
        "-",
        None,
        False,
        "Each UAV point to the horizontally closest stem (<= 8 m).",
    ),
    Method(
        "Watershed",
        "Canopy-height watershed",
        None,
        "-",
        None,
        False,
        "Basins of the canopy surface around its peaks, matched to stems within 3 m.",
    ),
    Method(
        "G_none",
        "Stem shortest paths",
        "basal",
        "none",
        1.0,
        False,
        "Shortest paths from basal seeds; no field heights.",
    ),
    Method(
        "G_lower_050",
        "Stem paths + lower-tree height",
        "basal",
        "lower",
        0.5,
        False,
        "As above, lower-stratum trees capped at field height + 0.5 m.",
    ),
    Method(
        "T_none",
        "Stem paths + canopy peaks",
        "peaks",
        "none",
        1.0,
        False,
        "Basal seeds plus upper seeds from matched canopy peaks; no heights.",
    ),
    Method(
        "T_lower_100",
        "Peaks + lower-tree height",
        "peaks",
        "lower",
        1.0,
        False,
        "Peak-seeded paths; lower trees capped at field height + 1.0 m.",
    ),
    Method(
        "T_coarse2_mid",
        "Peaks + height classes",
        "peaks",
        "classes",
        0.5,
        False,
        "Peak-seeded paths; lower heights rounded to 2 m classes, + 0.5 m.",
    ),
    Method(
        "T_targeted_1",
        "Peaks + requested heights",
        "peaks",
        "requested",
        1.0,
        False,
        "Heights only for lower trees that seem to absorb the canopy (48 requests).",
    ),
    Method(
        "G_fallback",
        "Stem paths + height + wider seed window",
        "basal",
        "lower",
        0.5,
        True,
        "G_lower_050 with the 4.5 m seed window for unseeded trees and height screening.",
    ),
    Method(
        "T_fallback",
        "Peaks + height classes + wider seed window",
        "peaks",
        "classes",
        0.5,
        True,
        "T_coarse2_mid with the 4.5 m seed window and height screening.",
    ),
    Method(
        "Hybrid",
        "Combined by assignment margin",
        None,
        "-",
        None,
        True,
        "G_fallback and T_fallback merged point by point by margin (combine.py).",
    ),
]

BY_CODE = {m.code: m for m in METHODS}
