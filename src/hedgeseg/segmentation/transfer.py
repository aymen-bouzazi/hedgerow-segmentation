"""Pass tree identities from the ALS points to the UAV points.

Physical idea
-------------
Identity is established in the leaf-off ALS, which samples wood inside the
crown; the leaf-on UAV cloud describes the outer leaf surface. Foliage hangs
close to the wood that carries it, so each UAV point is scored for every tree
by its distance to that tree's nearest assigned ALS point, plus a small
preference for the nearer stem:

    score = d(point, tree's ALS points) + 0.25 x horizontal distance to stem

A tree needs >= 5 ALS points to take part. A UAV point may not go to a tree
whose ALS points are > 4 m away or whose stem is > 8 m away. These limits are
active: they exclude about half of all tree-point pairs within 8 m of a stem,
but change only ~0.26 % of winning assignments, because foliage that far
from one tree's wood is almost always nearer another tree's.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import constants as C
from .scoring import pick_lowest_cost


def transfer_to_uav(
    als_xyz, als_label, uav_xyz, tree_ids, stems_xy, stem_ground_z, height_cap, allowance
):
    """Label every UAV point; same height penalty as the ALS stage.

    Computed in blocks of points and only for trees within 8 m, which gives
    exactly the same result as scoring all pairs while using far less memory.
    Returns the dict of :func:`pick_lowest_cost` (``label``, ``margin``).
    """
    within_8m = cKDTree(uav_xyz[:, :2]).query_ball_point(stems_xy, C.TRANSFER_MAX_STEM)
    support = {}
    for i, tree in enumerate(tree_ids):
        own = als_xyz[als_label == tree]
        if len(own) >= C.TRANSFER_MIN_ALS_POINTS:
            support[i] = cKDTree(own)

    label = np.zeros(len(uav_xyz), np.int32)
    margin = np.zeros(len(uav_xyz), np.float32)
    for start in range(0, len(uav_xyz), C.TRANSFER_BLOCK):
        stop = min(start + C.TRANSFER_BLOCK, len(uav_xyz))
        block = np.full((len(tree_ids), stop - start), np.inf, np.float32)
        for i, tree_support in support.items():
            idx = np.asarray(within_8m[i], dtype=np.int64)
            idx = idx[(idx >= start) & (idx < stop)]
            if not len(idx):
                continue
            pts = uav_xyz[idx]
            to_support = tree_support.query(pts)[0]
            to_stem = np.linalg.norm(pts[:, :2] - stems_xy[i], axis=1)
            score = (to_support + C.TRANSFER_DISTANCE_WEIGHT * to_stem).astype(np.float32)
            score[(to_support > C.TRANSFER_MAX_SUPPORT) | (to_stem > C.TRANSFER_MAX_STEM)] = np.inf
            if np.isfinite(height_cap[i]):
                excess = np.maximum(pts[:, 2] - stem_ground_z[i] - height_cap[i] - allowance, 0)
                score += (C.HEIGHT_PENALTY_COEFFICIENT * excess**2).astype(np.float32)
            block[i, idx - start] = score
        result = pick_lowest_cost(block, tree_ids)
        label[start:stop] = result["label"]
        margin[start:stop] = result["margin"]
    return {"label": label, "margin": margin}
