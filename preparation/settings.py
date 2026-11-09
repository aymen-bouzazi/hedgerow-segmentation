"""Paths used by the preparation scripts, read from config/preparation.json.

Relative paths are resolved from the repository root. Copy
config/preparation.example.json to config/preparation.json and edit it; the
real file is ignored by git so your local paths never end up online.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_cfg_path = ROOT / "config" / "preparation.json"
if not _cfg_path.exists():
    raise SystemExit("Create config/preparation.json from config/preparation.example.json first.")
_cfg = json.loads(_cfg_path.read_text())


def _path(key):
    value = _cfg.get(key)
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


OUT = _path("work_dir")                      # intermediate files (ennezat_B1.npz, ennezat_dz.json, reports)
OUT.mkdir(parents=True, exist_ok=True)
ALS = _path("als_raw")                       # IGN LiDAR HD tile (COPC LAZ), leaf-off, Feb 2023
UAV = _path("uav_raw")                       # Metashape dense cloud (LAZ), leaf-on, May 2026
STEMS = str(_path("stems_offset_gpkg"))      # surveyed stems after the 0.85 m offset
INVENTORY = _path("inventory_clean_gpkg")    # final inventory with corrected tree numbers
OUTLINE = _path("hedge_outline_gpkg")        # mapped hedge polygon and centreline
CAMERA_STATIONS = _path("camera_stations_csv")
EXPORT_DIR = _path("export_dir")
