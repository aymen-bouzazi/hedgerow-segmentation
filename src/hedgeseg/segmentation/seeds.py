"""Basal seeds: the few ALS returns that anchor each tree at its stem.

Physical idea
-------------
A leaf-off laser pulse rarely hits a narrow vertical trunk, but where it does,
those returns sit within a short distance of the surveyed stem and spread
vertically. A *seed* is such a small upright group. Every path-based method
starts from the seeds, so a tree without a seed can only be reached through
other evidence (canopy peaks, or the wider window below).
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .scoring import horizontal_distance


def _seed_for_tree(tree, nearest_tree, distance, height, max_height):
    """Indices of seed points for one tree, or an empty array."""
    candidates = np.flatnonzero(
        (nearest_tree == tree)
        & (distance[tree] <= C.SEED_RADIUS)
        & (height >= C.SEED_MIN_HEIGHT)
        & (height <= max_height)
    )
    order = np.argsort(
        distance[tree, candidates] + C.SEED_HEIGHT_WEIGHT * height[candidates],
        kind="stable",
    )
    chosen = candidates[order[: C.SEED_MAX_POINTS]]
    upright = len(chosen) >= C.SEED_MIN_POINTS and np.ptp(height[chosen]) >= C.SEED_MIN_SPAN
    return chosen if upright else np.array([], dtype=int)


def basal_seeds(als_xyz, als_height, stems_xy, max_height=C.SEED_MAX_HEIGHT):
    """Seed point indices for every tree (list of arrays, one per tree).

    Each ALS point is first allocated to its horizontally nearest stem, so no
    point can belong to two seeds. Among the points allocated to a tree, those
    within ``SEED_RADIUS`` and inside the height window are ranked by
    distance + 0.05 x height (nearest and lowest first) and the first 12 kept.
    The group is accepted if it has >= 3 points spanning >= 0.25 m vertically.
    """
    distance = horizontal_distance(als_xyz, stems_xy)
    nearest_tree = np.argmin(distance, axis=0)
    return [
        _seed_for_tree(i, nearest_tree, distance, als_height, max_height)
        for i in range(len(stems_xy))
    ]


def add_wide_window_seeds(seeds, als_xyz, als_height, stems_xy):
    """Retry trees without a seed using a taller window (up to 4.5 m).

    Used by the "wider seed window" methods. Some stems have no returns in
    the 0.35-2 m trunk zone (a shrub hidden under a neighbour, a walnut whose
    lower trunk was missed); their nearest returns are higher up. Radius and
    other rules stay the same. Returns (new_seeds, used_wide_window flags).
    """
    distance = horizontal_distance(als_xyz, stems_xy)
    nearest_tree = np.argmin(distance, axis=0)
    seeds = list(seeds)
    used = np.zeros(len(stems_xy), bool)
    for i in range(len(stems_xy)):
        if len(seeds[i]):
            continue
        wide = _seed_for_tree(i, nearest_tree, distance, als_height, C.SEED_MAX_HEIGHT_WIDE)
        if len(wide):
            seeds[i] = wide
            used[i] = True
    return seeds, used
