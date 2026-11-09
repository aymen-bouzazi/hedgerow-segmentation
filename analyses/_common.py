"""Helpers shared by the sensitivity analyses (not part of the main pipeline)."""

import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hedgeseg import io  # noqa: E402
from hedgeseg.evaluation.context import image_domain, reference_inputs  # noqa: E402


class PhotoScorer:
    """Mean F1 of one UAV label vector against the photo outlines.

    Same definition as the main evaluation ('all_projected' basis): every
    assigned point inside the image counts, nothing is clipped to the outline,
    trees need >= 20 reference points.
    """

    def __init__(self, uav_xyz):
        _, cams, outlines, _ = reference_inputs()
        self.items = []
        for pid in sorted({a["photo_id"] for a in outlines}):
            cam = cams[pid]
            uv, depth = io.project(uav_xyz, cam)
            visible = image_domain(uv, depth, cam)
            for a in (a for a in outlines if a["photo_id"] == pid):
                inside = MplPath(np.array(a["uv"])).contains_points(uv) & visible
                if inside.sum() >= 20:
                    self.items.append((int(a["gpsID"]), visible, inside))

    def mean_f1(self, labels):
        scores = []
        for tree, visible, inside in self.items:
            pred = (labels == tree) & visible
            tp = np.count_nonzero(inside & pred)
            fp = np.count_nonzero(~inside & pred)
            fn = np.count_nonzero(inside & ~pred)
            scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
        return float(np.nanmean(scores))
