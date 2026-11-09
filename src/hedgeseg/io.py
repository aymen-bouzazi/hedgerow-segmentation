"""Paths and loaders shared by every stage.

The project root is the repository folder, or the folder named by the
environment variable ``HEDGESEG_ROOT`` (useful to keep data elsewhere).

    data/source/   inputs, never modified (clouds, inventory, cameras, outlines)
    data/working/  arrays built by ``run.py prepare`` and the label files
    results/       every table and figure produced by the pipeline
    viewers/       the offline HTML viewers built by ``run.py viewers``
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HEDGESEG_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "data" / "source"
WORKING = ROOT / "data" / "working"
RESULTS = ROOT / "results"

# The six single-assignment methods stored in six_method_labels_full.npz,
# then the three wider-window/combined methods stored in hybrid_labels.npz.
SIX_METHOD_CODES = [
    "G_none",
    "G_lower_050",
    "T_none",
    "T_lower_100",
    "T_coarse2_mid",
    "T_targeted_1",
]


def config() -> dict:
    """Analysis settings (config/project.json)."""
    path = Path(os.environ.get("HEDGESEG_CONFIG", ROOT / "config" / "project.json"))
    if not path.exists():
        raise FileNotFoundError(
            "Private study configuration is missing. Copy config/project.example.json "
            "to config/project.json and fill in the site parameters locally."
        )
    settings = json.loads(path.read_text())
    if settings.get("hedge_origin") is None or settings.get("hedge_basis") is None:
        raise ValueError("Set the real hedge_origin and hedge_basis in the private configuration.")
    return settings


def sha(path) -> str:
    """SHA-256 of a file, used in receipts so any output can be traced to its inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj) -> None:
    """Write a JSON receipt (numpy values converted), creating folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
        ),
        encoding="utf-8",
    )


def clouds() -> dict:
    """Analysis point set built by ``run.py prepare`` (see docs/DATA.md for every array)."""
    return dict(np.load(WORKING / "clouds_full.npz", allow_pickle=False))


def inventory() -> pd.DataFrame:
    """Surveyed trees: number, position, stratum, field heights, ground at the stem."""
    return pd.read_csv(WORKING / "inventory.csv")


def cameras() -> dict:
    """Aligned camera models of the side photographs, keyed 'season/filename'."""
    out = {}
    for season in ("march", "april", "july"):
        for cam in json.loads((SOURCE / "cameras" / f"cameras_{season}_full.json").read_text()):
            cam = dict(cam)
            cam["season"] = season
            out[season + "/" + cam["filename"]] = cam
    return out


def rotation(angles):
    """World-to-camera rotation from (yaw clockwise from north, pitch up, roll)."""
    yaw, pitch, roll = angles
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    down = -np.cross(right, forward)
    return np.array(
        [
            np.cos(roll) * right + np.sin(roll) * down,
            -np.sin(roll) * right + np.cos(roll) * down,
            forward,
        ]
    )


def project(xyz, camera):
    """Pinhole projection of world points into a photograph.

    Returns (pixel coordinates (n, 2), depth along the viewing axis (n,)).
    u = W/2 + f * q1/q3,  v = H/2 + f * q2/q3  with q = R (X - C).
    """
    q = (np.asarray(xyz) - camera["camera"]) @ rotation(camera["angles"]).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2, None] * camera["focal_px"] + [
            camera["width"] / 2,
            camera["height"] / 2,
        ]
    return uv, q[:, 2]


def labels() -> dict:
    """UAV-point labels of all eleven methods, keyed by code name."""
    six = dict(np.load(WORKING / "six_method_labels_full.npz"))
    hybrid = dict(np.load(WORKING / "hybrid_labels.npz"))
    out = {name: six[name + "_uav_label"] for name in SIX_METHOD_CODES}
    out.update(
        G_fallback=hybrid["G_uav_label"],
        T_fallback=hybrid["T_uav_label"],
        Hybrid=hybrid["hybrid_uav_label"],
    )
    baseline = WORKING / "baseline_labels.npz"
    if baseline.exists():
        out.update(dict(np.load(baseline)))
    return out


def annotations() -> dict:
    """Hand-drawn visible-crown outlines on the photographs."""
    return json.loads((SOURCE / "annotations" / "review.json").read_text())
