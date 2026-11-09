"""Canopy-height model, its peaks, and upper seeds for tall trees.

Physical idea
-------------
A tall tree whose lower trunk returns nothing to the laser can still be
anchored at the top: its crown usually forms a local high point of the
canopy surface seen by the UAV. A peak matched to an upper-stratum stem
gives that tree a second set of seeds high in its crown.

A peak is a *surface feature*, not a tree: peaks are ~2 m apart while upper
crowns are several metres wide, so one crown can carry two peaks. It is the
match to a surveyed stem that turns a peak into evidence about one tree.

The same surface, partitioned into basins around its peaks, is the
canopy-height watershed comparator (see baselines.py).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from . import constants as C


def canopy_surface_and_basins(
    local_xy, height, min_separation=C.PEAK_MIN_SEPARATION, cell=C.CHM_CELL, smooth=C.CHM_SMOOTH
):
    """Build the canopy-height model, find its peaks and watershed basins.

    Parameters
    ----------
    local_xy : (n, 2) UAV point coordinates in the hedge frame
        (along-hedge, across-hedge), metres.
    height : (n,) UAV height above ground.

    Steps: highest point per 0.25 m cell -> fill empty cells only within
    0.75 m of data -> Gaussian smoothing (0.5 m) -> local maxima >= 2 m high
    and >= 2 m apart -> marker watershed of the inverted surface.

    Returns a dict with the per-point basin number (``basin``, 0 = none),
    peak positions in the hedge frame (``peak_xy``), their smoothed heights
    (``peak_heights``) and the rasters for plotting.
    """
    origin = np.floor(local_xy.min(axis=0) / cell) * cell - cell
    cells = np.floor((local_xy - origin) / cell).astype(int)
    shape = tuple(cells.max(axis=0) + 2)

    surface = np.full(shape, -np.inf, dtype=np.float32)
    np.maximum.at(surface, (cells[:, 0], cells[:, 1]), height)
    observed = np.isfinite(surface)
    gap, nearest = distance_transform_edt(~observed, return_indices=True)
    filled = surface[tuple(nearest)]
    filled[gap * cell > C.CHM_FILL_MAX] = 0
    smoothed = gaussian_filter(filled, sigma=smooth / cell)

    canopy = (gap * cell <= C.CHM_FILL_MAX) & (smoothed >= C.PEAK_MIN_HEIGHT)
    peaks = peak_local_max(
        smoothed,
        min_distance=max(1, int(round(min_separation / cell))),
        threshold_abs=C.PEAK_MIN_HEIGHT,
        labels=canopy.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros(shape, dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    basins = watershed(-smoothed, markers, mask=canopy)
    return {
        "basin": basins[cells[:, 0], cells[:, 1]].astype(np.int32),
        "peak_xy": origin + (peaks + 0.5) * cell,
        "peak_heights": smoothed[tuple(peaks.T)],
        "surface": smoothed,
        "basins": basins,
        "origin": origin,
        "cell": cell,
    }


def match_peaks_to_stems(stems_xy, peaks_xy, max_distance):
    """One-to-one matching of stems to peaks, each pair within ``max_distance``.

    Uses the Hungarian algorithm on horizontal distance; pairs farther apart
    than ``max_distance`` are forbidden (cost 1e6) and a stem may stay
    unmatched (dummy columns, cost 1000). Returns a list of
    (stem_row, peak_index, distance).
    """
    distance = distance_matrix(stems_xy, peaks_xy)
    n = len(stems_xy)
    cost = np.column_stack(
        [np.where(distance <= max_distance, distance, 1e6), np.full((n, n), 1000.0)]
    )
    rows, cols = linear_sum_assignment(cost)
    return [
        (r, c, distance[r, c])
        for r, c in zip(rows, cols)
        if c < len(peaks_xy) and distance[r, c] <= max_distance
    ]


def peak_seeds(als_xyz, als_height, stems_xy, upper_rows, canopy, frame_basis, frame_origin):
    """Upper seeds for upper-stratum trees matched to a canopy peak.

    For each upper tree matched to a peak within 3 m: take ALS returns within
    1.25 m of the peak, define the local top as their 98th-percentile height,
    keep returns in the upper 1 m, rank by distance + 0.15 x depth below the
    top and keep up to 8. Accepted if >= 3 points and the ALS top is within
    4 m of the UAV peak height (the two sensors broadly agree).

    Returns {tree_row: seed indices} and a list of association records.
    """
    peaks_world = canopy["peak_xy"] @ frame_basis + frame_origin
    matches = match_peaks_to_stems(stems_xy[upper_rows], peaks_world, C.PEAK_MATCH_DISTANCE)
    seeds, records = {}, []
    for row, peak, distance in matches:
        tree = upper_rows[row]
        plan_distance = np.linalg.norm(als_xyz[:, :2] - peaks_world[peak], axis=1)
        near = np.flatnonzero(plan_distance <= C.UPPER_SEED_RADIUS)
        if not len(near):
            continue
        top = np.quantile(als_height[near], C.UPPER_SEED_TOP_QUANTILE)
        chosen = near[als_height[near] >= top - C.UPPER_SEED_DEPTH]
        chosen = chosen[
            np.argsort(
                plan_distance[chosen] + C.UPPER_SEED_HEIGHT_WEIGHT * (top - als_height[chosen])
            )[: C.UPPER_SEED_MAX_POINTS]
        ]
        accepted = bool(
            len(chosen) >= 3
            and abs(top - canopy["peak_heights"][peak]) <= C.UPPER_SEED_MAX_TOP_DIFFERENCE
        )
        if accepted:
            seeds[tree] = chosen
        records.append(
            {
                "tree_row": int(tree),
                "peak_id": int(peak + 1),
                "distance_m": float(distance),
                "used": accepted,
                "n_seeds": int(len(chosen)) if accepted else 0,
            }
        )
    return seeds, records


def add_peak_seeds(basal, peak):
    """Union of basal seeds and peak seeds, tree by tree."""
    return [
        np.unique(np.r_[basal[i], peak[i]]) if i in peak else basal[i].copy()
        for i in range(len(basal))
    ]
