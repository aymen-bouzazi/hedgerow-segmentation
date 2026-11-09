"""Stage ``prepare``: build the common analysis point set from the shared clouds.

Input  : data/source/clouds/ennezat_{ALS,UAV}_full.laz  (already corrected and
         thinned; see preparation/ for how they were made from the raw data)
         data/source/inventory/ennezat_inventory_full_clean.gpkg
         data/source/clouds/ennezat_hedge_outline.gpkg
Output : data/working/clouds_full.npz, inventory.csv, stem_positions_full.csv,
         reference_full.csv, results/qa/preparation.json

Which points are kept, and why (settings in config/project.json):

* vegetation classes only: ALS 3/4/5, UAV 3/5 (ground is kept separately as
  the terrain reference, from ALS class 2 only - photogrammetry cannot see
  soil under a hedge in leaf);
* height above ground >= 0.35 m: removes ground, litter and the grass/weed
  layer of the field margin;
* within 8 m of the hedge axis: contains the hedge (99 % of its leaf-on
  vegetation lies within 3.5 m) and excludes crops;
* UAV points below 2.5 m must lie inside the mapped hedge polygon: removes
  low crops and margin vegetation inside the corridor.

Result: 45,965 of 304,597 ALS returns and 362,075 of 926,500 UAV points.
No offset is ever applied here: the inputs are already in one reference.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import io


def read_inventory() -> pd.DataFrame:
    """Read the corrected inventory GeoPackage without needing GDAL."""
    path = io.SOURCE / "inventory" / "ennezat_inventory_full_clean.gpkg"
    with sqlite3.connect(path) as con:
        table = con.execute("select table_name from gpkg_contents").fetchone()[0]
        t = pd.read_sql_query(f'SELECT * FROM "{table}"', con).drop(columns=["geom"])
    assert t.gpsID.is_unique and (t.gpsID > 0).all(), (
        "Tree numbers must be distinct positive integers"
    )
    expected = io.config().get("expected_inventory_count")
    if expected is not None:
        assert len(t) == expected, f"Expected {expected} trees, found {len(t)}"
    return t.sort_values("gpsID")


def read_hedge_polygon():
    """The mapped hedge polygon used to clean low UAV points."""
    import geopandas as gpd

    path = io.SOURCE / "clouds" / "ennezat_hedge_outline.gpkg"
    for name in gpd.list_layers(path).name:
        layer = gpd.read_file(path, layer=name)
        if layer.geom_type.str.contains("Polygon").any():
            return layer.geometry.union_all()
    raise ValueError("The hedge outline file needs a polygon layer")


def select_points() -> dict:
    """Stream both LAZ files in chunks and keep the analysis points."""
    import laspy
    from shapely import contains_xy

    cfg = io.config()
    origin, basis = np.array(cfg["hedge_origin"]), np.array(cfg["hedge_basis"])
    polygon = read_hedge_polygon()
    out = {
        "frame_origin": origin,
        "frame_basis": basis,
        "east": np.array([1.0, 0.0]),
        "north": np.array([0.0, 1.0]),
    }
    audit = []
    for sensor in ["als", "uav"]:
        path = io.SOURCE / "clouds" / f"ennezat_{sensor.upper()}_full.laz"
        parts, row = {}, 0
        with laspy.open(path) as reader:
            dims = list(reader.header.point_format.dimension_names)
            if "hag" not in dims:
                raise ValueError("Input must carry height above ground ('hag'); see preparation/")
            for chunk in reader.chunk_iterator(100_000):
                xyz = np.c_[chunk.x, chunk.y, chunk.z]
                hag = np.asarray(chunk.hag)
                cls = np.asarray(chunk.classification)
                along_across = (xyz[:, :2] - origin) @ basis.T  # hedge frame
                keep = (
                    np.isin(cls, [3, 4, 5] if sensor == "als" else [3, 5])
                    & (hag >= cfg["minimum_hag_m"])
                    & (abs(along_across[:, 1]) <= cfg["corridor_half_width_m"])
                )
                if sensor == "uav":
                    keep &= (hag >= cfg["uav_outline_limit_below_m"]) | contains_xy(
                        polygon, xyz[:, 0], xyz[:, 1]
                    )
                source_index = (
                    np.asarray(chunk.source_index)
                    if "source_index" in dims
                    else np.arange(row, row + len(chunk))
                )
                values = {
                    "xyz": xyz[keep],
                    "hag": hag[keep],
                    "plot_sd": along_across[keep],
                    "local": along_across[keep],
                    "source_index": source_index[keep],
                }
                if sensor == "uav":
                    values["rgb"] = np.c_[chunk.red, chunk.green, chunk.blue][keep]
                    values["voxel_count"] = (
                        np.asarray(chunk.voxel_count)
                        if "voxel_count" in dims
                        else np.ones(len(chunk), int)
                    )[keep]
                else:
                    ground = cls == 2
                    parts.setdefault("ground_xy", []).append(xyz[ground, :2])
                    parts.setdefault("ground_z", []).append(xyz[ground, 2])
                for key, value in values.items():
                    parts.setdefault(f"{sensor}_{key}", []).append(value)
                row += len(chunk)
        for key, value in parts.items():
            out[key] = np.concatenate(value)
        audit.append(
            {
                "sensor": sensor,
                "input_points": row,
                "retained_vegetation": len(out[f"{sensor}_xyz"]),
                "sha256": io.sha(path),
            }
        )
    out["als_index"] = out["als_source_index"]
    out["uav_sample_index"] = np.arange(len(out["uav_xyz"]))
    return out, audit


def run() -> None:
    """Stage ``prepare``: select points, compute ground at each stem, write the working files."""
    if io.config()["uav_additional_shift_m"] != 0:
        raise ValueError("Extra vertical shifts are not allowed in scientific outputs")
    io.WORKING.mkdir(parents=True, exist_ok=True)
    trees = read_inventory()
    arrays, audit = select_points()
    np.savez_compressed(io.WORKING / "clouds_full.npz", **arrays)

    # ground under each stem: median of the 8 nearest ALS ground returns
    nearest = cKDTree(arrays["ground_xy"]).query(trees[["x", "y"]], k=8)[1]
    trees["ground_z"] = np.median(arrays["ground_z"][nearest], axis=1)
    outlined = {a["gpsID"] for a in io.annotations()["annotations"] if a["type"] == "visible_crown"}
    trees["has_outline"] = trees.gpsID.isin(outlined)
    trees.to_csv(io.WORKING / "inventory.csv", index=False)
    trees[["gpsID", "x", "y"]].to_csv(io.WORKING / "stem_positions_full.csv", index=False)
    trees[["gpsID", "stratum", "Hsup", "Hinf", "HLCE", "state"]].to_csv(
        io.WORKING / "reference_full.csv", index=False
    )
    receipt = {
        "inventory_rows": len(trees),
        "outlined_trees": int(trees.has_outline.sum()),
        "ALS_vegetation": len(arrays["als_xyz"]),
        "UAV_vegetation": len(arrays["uav_xyz"]),
        "inputs": audit,
        "cloud_sha256": io.sha(io.WORKING / "clouds_full.npz"),
    }
    io.write_json(io.RESULTS / "qa" / "preparation.json", receipt)
    print(receipt, flush=True)
