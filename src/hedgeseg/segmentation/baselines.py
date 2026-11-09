"""Two simple comparators that every other method must beat.

Nearest-stem proximity
    Each UAV point goes to the horizontally closest stem (within 8 m).
    The simplest rule that uses the surveyed stems.

Canopy-height watershed
    The standard surface approach: partition the canopy-height model into
    basins around its peaks, then match each basin to the nearest stem
    within 3 m (one-to-one). It can only see trees that reach the canopy
    surface; at Ennezat all 130 lower-stratum trees grow beneath it.
    This is a minimal surface partition, not the full Dalponte & Coomes
    (2016) pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import constants as C
from .canopy_peaks import canopy_surface_and_basins, match_peaks_to_stems


def nearest_stem(uav_xyz, stems_xy, tree_ids):
    """Each UAV point to the horizontally closest stem within 8 m."""
    distance, nearest = cKDTree(stems_xy).query(uav_xyz[:, :2])
    return np.where(distance <= C.TRANSFER_MAX_STEM, tree_ids[nearest], 0).astype(np.int32)


def canopy_watershed(uav_local, uav_height, stems_xy, tree_ids, frame_basis, frame_origin):
    """Return (labels, list of stem-basin matches)."""
    canopy = canopy_surface_and_basins(uav_local, uav_height)
    peaks_world = canopy["peak_xy"] @ frame_basis + frame_origin
    basin_to_tree = np.zeros(len(peaks_world) + 1, np.int32)
    matches = []
    for row, peak, distance in match_peaks_to_stems(
        stems_xy, peaks_world, C.WATERSHED_MATCH_DISTANCE
    ):
        basin_to_tree[peak + 1] = tree_ids[row]
        matches.append(
            {"tree": int(tree_ids[row]), "peak": int(peak + 1), "distance_m": float(distance)}
        )
    return basin_to_tree[canopy["basin"]], matches
