"""Shortest-path scores through the leaf-off ALS points.

Physical idea
-------------
Wood is continuous: a branch is connected to its trunk. If ALS returns are
treated as stepping stones, a point that can be reached from a tree's seed
through closely spaced returns is more plausibly part of that tree than one
reached only by jumping across empty space.

How it is computed
------------------
1. Link every pair of ALS points closer than ``LINK_MAX_LENGTH`` (0.9 m).
2. Give each link the cost  c(l) = l * (1 + (l / 0.6)^2).
   Three 0.3 m steps cost 1.125; one 0.9 m jump over the same distance costs
   2.925, so routes through densely sampled vegetation are preferred.
3. Add one virtual node per tree at its stem base, linked to its seed points
   by their straight-line distance.
4. Dijkstra's algorithm gives, for every point, the cheapest route from each
   tree's stem (the *path cost*).
5. Final score = 0.35 x horizontal distance to the stem + 0.65 x path cost.

What it is not
--------------
It is not branch reconstruction. At the selected link distance, returns
from neighbouring trees can form a single connected component. Trees
are separated by *competing costs* (like houses closest by road to one town),
not by connectivity. Vegetation returns are not verified wood.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import constants as C
from .scoring import horizontal_distance
from .seeds import basal_seeds


def link_cost(length):
    """Cost of one link of the given length (m)."""
    return length * (1 + (length / C.LINK_COST_SCALE) ** 2)


def stem_path_scores(
    als_xyz, als_height, stems_xy, stem_ground_z, seeds=None, max_link=C.LINK_MAX_LENGTH
):
    """Score every ALS point against every tree with the path rule.

    Parameters
    ----------
    als_xyz : (n, 3) ALS coordinates (m, Lambert-93 / NGF-IGN69).
    als_height : (n,) height above ground of each point (m).
    stems_xy : (t, 2) surveyed stem positions.
    stem_ground_z : (t,) ground elevation at each stem (m).
    seeds : list of index arrays, one per tree; computed if not given.
    max_link : maximum link length (m); exposed for sensitivity studies.

    Returns
    -------
    scores : (t, n) array, lower = more plausible owner.
    seeds : the seeds used.
    diagnostics : dict (number of links, connected pieces, reachable share...).
    """
    n_points, n_trees = len(als_xyz), len(stems_xy)
    if seeds is None:
        seeds = basal_seeds(als_xyz, als_height, stems_xy)

    # 1-2. links between nearby points, with their cost (both directions)
    pairs = cKDTree(als_xyz).query_pairs(max_link, output_type="ndarray")
    a, b = pairs.T
    length = np.maximum(np.linalg.norm(als_xyz[b] - als_xyz[a], axis=1), C.LINK_MIN_LENGTH)
    cost = link_cost(length)
    rows, cols = np.r_[a, b], np.r_[b, a]
    weights = np.r_[cost, cost]

    point_graph = coo_matrix((weights, (rows, cols)), shape=(n_points, n_points)).tocsr()
    n_pieces, piece = connected_components(point_graph, directed=False)

    # 3. one virtual stem-base node per tree, linked to its seeds
    for tree, seed in enumerate(seeds):
        if len(seed):
            base = np.r_[stems_xy[tree], stem_ground_z[tree]]
            rows = np.r_[rows, np.full(len(seed), n_points + tree)]
            cols = np.r_[cols, seed]
            weights = np.r_[weights, np.linalg.norm(als_xyz[seed] - base, axis=1)]
    graph = coo_matrix(
        (weights, (rows, cols)), shape=(n_points + n_trees, n_points + n_trees)
    ).tocsr()

    # 4. cheapest route from every stem base to every point
    path_cost = dijkstra(graph, directed=True, indices=np.arange(n_points, n_points + n_trees))
    path_cost = path_cost[:, :n_points]

    # 5. combine with plan-view distance
    scores = C.WEIGHT_DISTANCE * horizontal_distance(als_xyz, stems_xy) + C.WEIGHT_PATH * path_cost
    diagnostics = {
        "max_link_m": max_link,
        "seeds_usable": sum(bool(len(s)) for s in seeds),
        "graph_links": len(pairs),
        "connected_pieces": int(n_pieces),
        "largest_piece_fraction": float(np.bincount(piece).max() / n_points),
        "reachable_fraction": float(np.isfinite(path_cost).any(axis=0).mean()),
    }
    return scores, seeds, diagnostics
