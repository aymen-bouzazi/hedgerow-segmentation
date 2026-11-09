"""Stable-surface residual audit; never infer a translation from changing crowns."""

import laspy
import numpy as np
import pandas as pd

from ..io import ROOT, write_json


def road_cells(sensor, classes):
    """Median elevation per grid cell of road-class points from one shared cloud."""
    parts = []
    with laspy.open(ROOT / f"data/source/clouds/ennezat_{sensor}_full.laz") as f:
        for p in f.chunk_iterator(100000):
            k = np.isin(p.classification, classes)
            parts.append(np.c_[np.asarray(p.x)[k], np.asarray(p.y)[k], np.asarray(p.z)[k]])
    a = np.concatenate(parts)
    d = pd.DataFrame(
        {"ix": np.floor(a[:, 0]).astype(int), "iy": np.floor(a[:, 1]).astype(int), "z": a[:, 2]}
    )
    return (
        d.groupby(["ix", "iy"])
        .z.agg(
            n="size",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def run():
    """Stage ``road-residuals``: drone-minus-laser height on the road after correction (should be ~0)."""
    a = road_cells("ALS", [2])
    u = road_cells("UAV", [11])
    t = u.merge(a, on=["ix", "iy"], suffixes=("_uav", "_als"))
    t["dz_uav_minus_als"] = t.median_uav - t.median_als
    t.to_csv(ROOT / "results/qa/registration_road_cells.csv", index=False)
    summary = []
    for name, k in [
        ("all_common_cells", np.ones(len(t), bool)),
        ("at_least_3_each", (t.n_uav >= 3) & (t.n_als >= 3)),
        (
            "low_roughness",
            (t.n_uav >= 3)
            & (t.n_als >= 3)
            & ((t.q75_uav - t.q25_uav) <= 0.15)
            & ((t.q75_als - t.q25_als) <= 0.15),
        ),
    ]:
        v = t.loc[k, "dz_uav_minus_als"]
        summary.append(
            {
                "selection": name,
                "cells": len(v),
                "median_m": float(v.median()),
                "iqr_m": float(v.quantile(0.75) - v.quantile(0.25)),
            }
        )
    write_json(
        ROOT / "results/qa/registration.json",
        {
            "surface": "UAV class 11 intersect ALS class 2; 1 m cells",
            "summaries": summary,
            "extra_correction_applied": False,
            "horizontal_registration_established": False,
            "interpretation": "Residual diagnostic on supplied already corrected clouds. Selection, roughness, georeferencing and dates affect it. It is not a new calibrated correction plane.",
        },
    )
    print(summary, flush=True)
