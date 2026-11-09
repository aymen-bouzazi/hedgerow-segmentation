"""Stage ``check``: consistency checks on everything the pipeline produced.

Inputs exist and agree (IDs, point counts, cameras), labels are valid,
scores are bounded, crown parameters are physically ordered (HCB < HLCE <
Htot, radii > 0), viewers exist if built. Fails loudly on any violation.
"""

import numpy as np
import pandas as pd
from PIL import Image

from .io import (
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
from .viewers.photos import DISPLAY, check_display


def run():
    """Stage ``check``: run every consistency check and stop on the first failure list."""
    tests = []

    def check(name, value):
        tests.append({"check": name, "passed": bool(value)})

    cfg = config()
    d = clouds()
    inv = inventory()
    ls = labels()
    cs = cameras()
    ann = annotations()
    check("unique inventory IDs", inv.gpsID.is_unique)
    if cfg.get("expected_inventory_count") is not None:
        check("expected inventory population", len(inv) == cfg["expected_inventory_count"])
    check("ID165 retained", 165 in inv.gpsID.values)
    for sensor in ["als", "uav"]:
        xyz = d[sensor + "_xyz"]
        h = d[sensor + "_hag"]
        sd = (xyz[:, :2] - cfg["hedge_origin"]) @ np.array(cfg["hedge_basis"]).T
        check(sensor + " XYZ finite", np.isfinite(xyz).all())
        check(sensor + " point HAG correspondence", len(xyz) == len(h))
        check(sensor + " minimum HAG", h.min() >= cfg["minimum_hag_m"] - 1e-6)
        check(sensor + " corridor", np.max(abs(sd[:, 1])) <= cfg["corridor_half_width_m"] + 1e-8)
    check("unique UAV source IDs", len(np.unique(d["uav_source_index"])) == len(d["uav_xyz"]))
    for name, y in ls.items():
        check(name + " label length", len(y) == len(d["uav_xyz"]))
        check(name + " IDs exist", np.isin(y, np.r_[0, inv.gpsID]).all())
    h = np.load(ROOT / "data/working/hybrid_labels.npz")
    s = np.load(ROOT / "data/working/six_method_labels_full.npz")
    check(
        "duplicated hybrid label refreshed",
        np.array_equal(h["hybrid_uav_label"], s["HYBRID_uav_label"]),
    )
    for sensor in ["als", "uav"]:
        union = (h["G_" + sensor + "_label"] > 0) | (h["T_" + sensor + "_label"] > 0)
        check(
            sensor + " sole-parent Hybrid coverage preserved",
            np.array_equal(union, h["hybrid_" + sensor + "_label"] > 0),
        )
    missing_originals = []
    for pid, c in cs.items():
        R = rotation(c["angles"])
        check(pid + " orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-10))
        original = ROOT / "data/source/photos" / pid
        if original.exists():
            with Image.open(original) as im:
                check(pid + " original native size", list(im.size) == [c["width"], c["height"]])
        else:
            missing_originals.append(pid)
        check_display(pid, c, DISPLAY / pid)
        check(pid + " verified display photograph", True)
    crowns = [a for a in ann["annotations"] if a["type"] == "visible_crown"]
    if cfg.get("expected_reference_count") is not None:
        check("expected reference population", len(crowns) == cfg["expected_reference_count"])
    check("one primary reference per individual", len({a["gpsID"] for a in crowns}) == len(crowns))
    check("all reference IDs exist", all(a["gpsID"] in inv.gpsID.values for a in crowns))
    check(
        "no duplicate reference per photo and tree",
        len(set((a["photo_id"], a["gpsID"]) for a in crowns)) == len(crowns),
    )
    scores = pd.read_csv(ROOT / "results/tables/evaluation_per_tree.csv")
    base = scores.query("metric_basis=='all_projected'")
    check("one evaluation per method and reference", len(base) == len(ls) * len(crowns))
    check(
        "F1 and IoU bounded",
        ((scores[["f1", "iou"]].fillna(0) >= 0) & (scores[["f1", "iou"]].fillna(0) <= 1))
        .all()
        .all(),
    )
    check("false positives retained", base.fp.sum() > 0)
    p = pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
    fit = p[p.status.str.startswith("model_fit")]
    check("parameter population", len(p) == len(inv) * len(ls))
    check("crown order", ((fit.hcb >= 0) & (fit.hcb < fit.hlce) & (fit.hlce < fit.htot)).all())
    check("radii positive", (fit[["rnorth", "reast", "rsouth", "rwest"]] > 0).all().all())
    if (ROOT / "viewers" / "index.html").exists():
        for name in [
            "index.html",
            "review_full_hedgerow.html",
            "inspect_trees_3D.html",
            "assets/data.js",
            "assets/shapes.js",
            "assets/three.min.js",
        ]:
            check("viewer " + name, (ROOT / "viewers" / name).is_file())
    write_json(
        ROOT / "results/qa/integration_checks.json",
        {
            "checks": tests,
            "passed": sum(t["passed"] for t in tests),
            "total": len(tests),
            "full_resolution_photos_unavailable": missing_originals,
            "scope": "Chapter 2; display verification does not verify absent original bytes; simulation checks excluded",
        },
    )
    failures = [t["check"] for t in tests if not t["passed"]]
    if failures:
        raise AssertionError("; ".join(failures))
    print(f"{len(tests)}/{len(tests)} integration checks passed", flush=True)
