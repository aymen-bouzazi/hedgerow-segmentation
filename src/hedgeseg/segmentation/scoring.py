"""Turn per-tree costs into one label per point.

Every assignment method in this package produces a *cost matrix*: one row per
surveyed tree, one column per point, lower = more plausible owner. This module
holds the two small operations shared by all methods.
"""

from __future__ import annotations

import numpy as np


def horizontal_distance(xyz: np.ndarray, stems_xy: np.ndarray) -> np.ndarray:
    """Horizontal distance (m) from every point to every stem.

    Returns an array of shape (n_trees, n_points), float32.
    Physical meaning: how far a point lies from a trunk in plan view, ignoring
    height. It is the whole of the nearest-stem method and a secondary term in
    the others.
    """
    return np.linalg.norm(xyz[None, :, :2] - stems_xy[:, None, :], axis=2).astype(np.float32)


def pick_lowest_cost(costs: np.ndarray, tree_ids: np.ndarray) -> dict:
    """Give each point to the tree with the lowest cost.

    Parameters
    ----------
    costs : (n_trees, n_points) array. ``inf`` means "this tree may not own
        this point" (e.g. outside its distance limits).
    tree_ids : (n_trees,) surveyed tree numbers, in the same order as rows.

    Returns
    -------
    dict with
        ``label``  : winning tree number per point, 0 when no tree is feasible.
        ``margin`` : cost of the second-best tree minus the best one. A large
                     margin means the decision was clear-cut. It is a cost
                     separation, NOT a probability. A point with a single
                     feasible tree gets margin 0 (no competing evidence).
    """
    costs = np.asarray(costs, dtype=np.float32)
    columns = np.arange(costs.shape[1])
    best_row = np.argmin(costs, axis=0)
    best = costs[best_row, columns]
    if len(tree_ids) > 1:
        second = np.partition(costs, 1, axis=0)[1]
    else:
        second = np.full(len(best), np.inf)
    with np.errstate(invalid="ignore"):
        margin = second - best
    margin[~np.isfinite(margin)] = 0
    label = np.where(np.isfinite(best), tree_ids[best_row], 0).astype(np.int32)
    return {"label": label, "margin": margin.astype(np.float32)}
