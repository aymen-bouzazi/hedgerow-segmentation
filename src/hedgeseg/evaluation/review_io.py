"""Validate portable reviews and select primary references for evaluation.

Camera changes are orientation-only relative to the supplied models. A review
cannot move points, stems, camera centres or native image coordinates.
"""

import copy
import json
from pathlib import Path

import numpy as np

from ..io import ROOT, annotations, cameras, inventory, sha


def context(review_path=None):
    """Load and validate a review file (outlines + accepted camera orientations); never moves points or cameras."""
    supplied = cameras()
    source = (
        Path(review_path).resolve() if review_path else ROOT / "data/source/annotations/review.json"
    )
    review = json.loads(source.read_text())
    known_ids = set(inventory().gpsID.astype(int))
    if not isinstance(review.get("annotations"), list):
        raise ValueError("Review needs an annotations array")
    native_declarations = [
        "native_image_pixels",
        "original image pixels; origin top-left; x right, y down",
    ]
    if review.get("coordinate_system", "native_image_pixels") not in native_declarations:
        raise ValueError("Review observations must be in native image pixels")
    if review.get("uav_z_shift_m", 0) != 0 or review.get("registration_applied_in_view", False):
        raise ValueError(
            "Legacy coordinate-shift previews cannot be evaluated. Import and export with the restored tool first."
        )
    photo_meta = {p["photo_id"]: p for p in annotations().get("photos", [])}
    seen = set()
    for a in review["annotations"]:
        pid = a.get("photo_id")
        camera = supplied.get(pid)
        uid = a.get("uid")
        gps = a.get("gpsID")
        if camera is None or not uid or uid in seen or type(gps) is not int or gps not in known_ids:
            raise ValueError("Unknown photograph/ID or duplicate annotation UID: " + str(uid))
        seen.add(uid)
        uv = np.asarray(a.get("uv", []), dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
            raise ValueError("Invalid observation coordinates: " + uid)
        if (
            (uv < 0).any()
            or (uv[:, 0] >= camera["width"]).any()
            or (uv[:, 1] >= camera["height"]).any()
        ):
            raise ValueError("Observation outside native frame: " + uid)
        if a.get("type") == "visible_crown" and (len(uv) < 3 or len(np.unique(uv, axis=0)) < 3):
            raise ValueError("Crown polygon requires three distinct vertices: " + uid)
        if a.get("type") in ["visible_ground_base", "stem_base"] and len(uv) != 1:
            raise ValueError("Base must have one point: " + uid)
        if [a.get("native_width", camera["width"]), a.get("native_height", camera["height"])] != [
            camera["width"],
            camera["height"],
        ]:
            raise ValueError("Annotation/camera native dimensions disagree: " + uid)
        original = photo_meta.get(pid, {}).get("original_sha256")
        if a.get("original_sha256") and original and a["original_sha256"] != original:
            raise ValueError("Original photograph hash mismatch: " + uid)
    effective = copy.deepcopy(supplied)
    changed = []

    def accept(pid, camera):
        if pid not in supplied or not isinstance(camera, dict):
            raise ValueError("Unknown portable camera: " + str(pid))
        base = supplied[pid]
        if not all(camera.get(k) == base[k] for k in ["camera", "focal_px", "width", "height"]):
            raise ValueError(
                "Camera revision must preserve supplied centre, focal length and dimensions: " + pid
            )
        angles = np.asarray(camera.get("angles", []), dtype=float)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("Invalid camera orientation: " + pid)
        effective[pid] = dict(base, angles=angles.tolist())

    models = review.get("camera_models", {})
    if not isinstance(models, dict):
        raise ValueError("camera_models must be an object")
    for pid, camera in models.items():
        accept(pid, camera)
    overrides = review.get("camera_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("camera_overrides must be an object")
    for pid, revision in overrides.items():
        if not isinstance(revision, dict) or not isinstance(revision.get("camera"), dict):
            raise ValueError("Camera override needs a camera object: " + pid)
        camera = revision["camera"]
        if pid in models and camera.get("angles") != models[pid].get("angles"):
            raise ValueError("Conflicting accepted camera model and override: " + pid)
        accept(pid, camera)
    for pid, camera in effective.items():
        if camera["angles"] != supplied[pid]["angles"]:
            changed.append(pid)
    outlines = [a for a in review["annotations"] if a.get("type") == "visible_crown"]
    primary = review.get("primary_reference_uids", {})
    if not isinstance(primary, dict):
        raise ValueError("primary_reference_uids must map surveyed IDs to annotation UIDs")
    for gps, uid in primary.items():
        if not any(str(a["gpsID"]) == str(gps) and a["uid"] == uid for a in outlines):
            raise ValueError("Primary UID does not identify a crown of ID " + str(gps))
    chosen = []
    for gps in sorted({a["gpsID"] for a in outlines}):
        group = [a for a in outlines if a["gpsID"] == gps]
        uid = primary.get(str(gps))
        if uid is not None:
            group = [a for a in group if a["uid"] == uid]
            if len(group) != 1:
                raise ValueError("Primary UID does not identify a crown of ID " + str(gps))
        elif len(group) > 1:
            raise ValueError(
                "Select one primary_reference_uids entry for duplicated crown ID " + str(gps)
            )
        chosen.extend(group)
    if not chosen:
        raise ValueError("No primary visible-crown references")
    receipt = {
        "review_path": str(source),
        "review_sha256": sha(source),
        "primary_reference_count": len(chosen),
        "all_crown_observations": len(outlines),
        "camera_models_changed": sorted(changed),
        "missing_reference_ids": sorted(known_ids - {a["gpsID"] for a in chosen}),
        "camera_override_count": len(overrides),
        "primary_reference_policy": "one explicitly selected visible-crown outline per individual; no automatic multi-view pooling",
        "source_coordinates_changed": False,
        "offsets_reapplied": False,
        "source_camera_hashes": {
            p.name: sha(p) for p in (ROOT / "data/source/cameras").glob("*.json")
        },
    }
    return review, effective, chosen, receipt
