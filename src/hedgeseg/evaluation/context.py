"""Shared accepted references and image domain for all photographic diagnostics."""

from pathlib import Path

import numpy as np

from ..io import ROOT, config, project
from .review_io import context


def image_domain(uv, depth, camera):
    """Points that project inside the photograph and in front of the camera."""
    return (
        (depth > 0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera["width"])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera["height"])
    )


def reference_inputs(review_path=None):
    """Validated outlines (one per tree), cameras and a receipt describing them."""
    ann, cameras, outlines, receipt = context(review_path)
    exclusions = config().get("manual_exclusions", {}) or {}
    if not isinstance(exclusions, dict) or any(not str(v).strip() for v in exclusions.values()):
        raise ValueError("manual_exclusions must map IDs to nonempty reasons")
    excluded = {int(k): str(v) for k, v in exclusions.items()}
    selected = [a for a in outlines if a["gpsID"] not in excluded]
    if not selected:
        raise ValueError("No scorable primary references after explicit exclusions")
    receipt = dict(
        receipt,
        manually_excluded_ids=sorted(excluded),
        manual_exclusion_reasons=excluded,
        scored_primary_reference_count=len(selected),
        prediction_domain="positive camera depth, finite projection, native image rectangle",
        reference_clipping_of_predictions=False,
    )
    return ann, cameras, selected, receipt


def projected_frames(xyz, cameras, outlines):
    """Yield, photo by photo, the projected UAV points and the outlines drawn on it."""
    for pid in sorted({a["photo_id"] for a in outlines}):
        c = cameras[pid]
        uv, depth = project(xyz, c)
        yield (
            pid,
            c,
            uv,
            depth,
            image_domain(uv, depth, c),
            [a for a in outlines if a["photo_id"] == pid],
        )


def result_root(output_dir=None):
    """Results folder (default ``results/``), created if needed."""
    dest = Path(output_dir).resolve() if output_dir else ROOT / "results"
    for name in ("tables", "qa"):
        (dest / name).mkdir(parents=True, exist_ok=True)
    return dest
