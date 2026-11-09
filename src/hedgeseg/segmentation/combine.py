"""Merge two assignments point by point (the 'combined' method).

The two wider-seed-window methods are complementary: the stem-path one is
stronger in the lower stratum, the peak-seeded one in the upper stratum.

Rule:
* a point labelled by only one parent keeps that label;
* where both label it, the parent with the larger assignment margin wins;
* ties go to the stem-path parent.

Margins are cost separations, not probabilities; the rule is an explicit
policy and does not look at the photographs.
"""

from __future__ import annotations

import numpy as np


def combine_by_margin(path_label, peak_label, path_margin, peak_margin):
    """Return (combined_label, took_stem_path_parent)."""
    g, t = np.asarray(path_label), np.asarray(peak_label)
    gm = np.nan_to_num(np.asarray(path_margin), nan=-np.inf)
    tm = np.nan_to_num(np.asarray(peak_margin), nan=-np.inf)
    if not (g.shape == t.shape == gm.shape == tm.shape):
        raise ValueError("Parent labels and margins must have identical shapes")
    take_path = (g > 0) & ((t == 0) | (gm >= tm))
    return np.where(take_path, g, t), take_path
