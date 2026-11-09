"""Field heights as an upper bound on what a tree can own.

Physical idea
-------------
A shortest path has no notion of height. A suppressed 3.7 m shrub under a
13 m walnut is connected to the walnut's lower crown, and can "grow" into it
in the assignment. A field height gives the missing biological bound: a
3.7 m shrub cannot own a point at 12 m.

The penalty is one-sided (a tree may look shorter in the cloud than in the
field, but cannot own points far above itself) and soft (it discourages, but
does not forbid, such ownership):

    P = 2 * max(Z - g - H_field - a, 0)^2

with Z the point elevation, g the ground at the stem, H_field the surveyed
height and a an allowance (0.5 or 1.0 m) that also absorbs the difference
between survey date and acquisition date.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def height_penalty(xyz, stem_ground_z, height_cap, allowance):
    """(n_trees, n_points) penalty; zero for trees whose cap is NaN."""
    penalty = (
        C.HEIGHT_PENALTY_COEFFICIENT
        * np.maximum(xyz[None, :, 2] - stem_ground_z[:, None] - height_cap[:, None] - allowance, 0)
        ** 2
    )
    penalty[~np.isfinite(height_cap)] = 0
    return penalty.astype(np.float32)


def height_classes(heights):
    """Round heights to the nearest 2 m class, never below 0.5 m.

    Tests whether coarse field information (shrub vs. small tree vs. tall
    tree) is enough for the constraint to work.
    """
    return np.maximum(
        C.HEIGHT_CLASS_MINIMUM, np.floor(heights / C.HEIGHT_CLASS + 0.5) * C.HEIGHT_CLASS
    )


def selective_requests(
    uav_label, uav_xyz, uav_along, stems_along, stem_ground_z, tree_ids, is_lower
):
    """Which lower trees would need a field height (the 'requested heights' variant).

    A lower tree is flagged when its assigned top (99.5th percentile, >= 20
    points) reaches within 1 m of the local canopy envelope (same percentile
    of all UAV points within 1.5 m along the hedge), i.e. it looks as if it
    has absorbed the overstorey, or when its top cannot be estimated.
    """
    tops, envelope = [], []
    for j, tree in enumerate(tree_ids):
        own = uav_label == tree
        tops.append(
            np.quantile(uav_xyz[own, 2] - stem_ground_z[j], C.TOP_QUANTILE)
            if own.sum() >= C.MIN_POINTS_FOR_TOP
            else np.nan
        )
        local = np.abs(uav_along - stems_along[j]) <= C.REQUEST_WINDOW
        envelope.append(
            np.quantile(uav_xyz[local, 2] - stem_ground_z[j], C.TOP_QUANTILE)
            if local.sum() >= C.MIN_POINTS_FOR_TOP
            else np.nan
        )
    tops, envelope = np.asarray(tops), np.asarray(envelope)
    return is_lower & ((tops >= envelope - C.REQUEST_MARGIN) | ~np.isfinite(tops))
