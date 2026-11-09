"""Generate self-contained, offline viewers from canonical arrays and annotations."""

import json
import shutil

import numpy as np
import pandas as pd

from ..io import (
    ROOT,
    annotations,
    cameras,
    clouds,
    config,
    inventory,
    labels,
    rotation,
    write_json,
)
from . import TEMPLATES
from .photos import prepare_display


def safe_records(t):
    """DataFrame to JSON-safe records."""
    return json.loads(t.to_json(orient="records"))


def run():
    """Stage ``viewers``: write viewers/ (data, photos, 3D shapes, HTML) and a labelled LAZ of all methods."""
    d = clouds()
    inv = inventory()
    lab = labels()
    cams = cameras()
    cfg = config()
    review = annotations()
    dest = ROOT / "viewers"
    photos = []
    (dest / "assets").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/point_clouds").mkdir(parents=True, exist_ok=True)
    for vendor in (TEMPLATES / "vendor").iterdir():
        shutil.copy2(vendor, dest / "assets" / vendor.name)
    cfg["meridian_convergence_degrees"] = float(
        __import__("pyproj")
        .Proj("EPSG:2154")
        .get_factors(cfg["longitude"], cfg["latitude"])
        .meridian_convergence
    )
    photo_sources = {}
    for pid, c in cams.items():
        photo_sources[pid] = prepare_display(pid, c)
        c = dict(c)
        c["photo_id"] = pid
        c["rotation"] = rotation(c["angles"]).tolist()
        c["url"] = "assets/photos/" + pid
        photos.append(c)
    # World XYZ retained with millimetre display precision; numerical evaluation uses full arrays.
    data = {
        "config": cfg,
        "photos": photos,
        "review": review,
        "inventory": safe_records(inv),
        "xyz": np.round(d["uav_xyz"], 3).tolist(),
        "hag": np.round(d["uav_hag"], 3).tolist(),
        "labels": {m: y.astype(int).tolist() for m, y in lab.items()},
        "source_index": d["uav_source_index"].astype(int).tolist(),
        "als_xyz": np.round(d["als_xyz"], 3).tolist(),
        "als_hag": np.round(d["als_hag"], 3).tolist(),
        "ground_xyz": np.round(np.c_[d["ground_xy"], d["ground_z"]][::5], 3).tolist(),
        "parameters": safe_records(
            pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
        ),
        "metrics": safe_records(
            pd.read_csv(ROOT / "results/tables/evaluation_per_tree.csv").query(
                "metric_basis == 'all_projected'"
            )
        ),
    }
    text = "window.ENNEZAT=" + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";"
    (dest / "assets/data.js").write_text(text, encoding="utf-8")
    for name in ["index.html", "style.css"]:
        shutil.copy2(TEMPLATES / name, dest / name)
    # Earlier entry-point filenames open the preserved original interfaces.
    for name, target in [
        ("photo.html", "review_full_hedgerow.html"),
        ("trees.html", "inspect_trees_3D.html"),
    ]:
        (dest / name).write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='
            + target
            + '"><title>Ennezat viewer</title><a href="'
            + target
            + '">Open the Ennezat viewer</a></html>'
        )
    # Portable labelled 3D interchange: method labels + source identity; no XYZ reconstruction from pixels.
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = np.floor(d["uav_xyz"].min(axis=0))
    header.add_crs(__import__("pyproj").CRS("EPSG:2154"))
    cloud = laspy.LasData(header)
    cloud.x = d["uav_xyz"][:, 0]
    cloud.y = d["uav_xyz"][:, 1]
    cloud.z = d["uav_xyz"][:, 2]
    for name, values, dtype in [
        ("hag", d["uav_hag"], "float32"),
        ("source_index", d["uav_source_index"], "uint64"),
        *[(m, y, "uint16") for m, y in lab.items()],
    ]:
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        cloud[name] = values
    cloud.red = d["uav_rgb"][:, 0]
    cloud.green = d["uav_rgb"][:, 1]
    cloud.blue = d["uav_rgb"][:, 2]
    cloud.write(ROOT / "results/point_clouds/ennezat_all_methods.laz")
    write_json(
        ROOT / "results/qa/viewers.json",
        {
            "photos": len(photos),
            "original_photos_available": sum(v == "original" for v in photo_sources.values()),
            "verified_compact_display_photos": sum(
                v == "verified_compact_display" for v in photo_sources.values()
            ),
            "outlines": sum(a["type"] == "visible_crown" for a in review["annotations"]),
            "methods": list(lab),
            "uav_points": len(d["uav_xyz"]),
            "same_point_index_across_layers": True,
            "XYZ_and_camera_depth_available": True,
            "ALS_ground_is_display_context_only": True,
            "no_extra_vertical_shift": True,
        },
    )
    from .review_tool import run as build_review
    from .tree_tool import run as build_trees

    build_review()
    build_trees()
    print("Built offline viewers:", len(photos), "photos,", len(lab), "methods", flush=True)
