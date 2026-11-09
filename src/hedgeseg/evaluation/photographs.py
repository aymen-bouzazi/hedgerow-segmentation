"""Projection agreement, never interpreted as ground-truth 3D segmentation accuracy."""

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from shapely import contains_xy
from shapely.geometry import Polygon

from ..io import ROOT, clouds, config, inventory, labels, project, write_json
from .context import image_domain, reference_inputs, result_root


def counts(truth, pred, valid):
    """TP, FP, FN, precision, recall, F1 and IoU for one tree in one photograph."""
    tp = int(np.count_nonzero(truth & pred & valid))
    fp = int(np.count_nonzero(~truth & pred & valid))
    fn = int(np.count_nonzero(truth & ~pred & valid))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
    }


def front_mask(uv, depth, visible, width, cell=12, tolerance=0.4):
    """Nearest available return per image cell, with a depth slab. A sensitivity filter only."""
    q = np.flatnonzero(visible)
    ncol = int(np.ceil(width / cell)) + 1
    if not len(q):
        return np.zeros(len(depth), bool)
    ix = np.floor(uv[q, 1] / cell).astype(int) * ncol + np.floor(uv[q, 0] / cell).astype(int)
    _, inv = np.unique(ix, return_inverse=True)
    nearest = np.full(inv.max() + 1, np.inf)
    np.minimum.at(nearest, inv, depth[q])
    mask = np.zeros(len(depth), bool)
    mask[q] = depth[q] <= nearest[inv] + tolerance
    return mask


def run(review_path=None, output_dir=None):
    """Stage ``evaluate``: score every method against every outline, write per-tree, summary and paired-difference tables."""
    from ..io import sha

    cfg = config()
    d = clouds()
    lab = labels()
    inv = inventory().set_index("gpsID")
    ann, cams, outl, review_receipt = reference_inputs(review_path)
    dest = result_root(output_dir)
    rows = []
    qa = []
    if not outl:
        raise ValueError("No visible_crown reference annotations were provided.")
    ids = [a["gpsID"] for a in outl]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Select one primary reference per individual or implement an explicit multi-view aggregation policy before scoring."
        )
    groups = {
        p: [a for a in outl if a["photo_id"] == p] for p in sorted({a["photo_id"] for a in outl})
    }
    for pid, aa in groups.items():
        c = cams[pid]
        uv, z = project(d["uav_xyz"], c)
        vis = image_domain(uv, z, c)
        front = front_mask(
            uv, z, vis, c["width"], cfg["depth_cell_native_px"], cfg["depth_tolerance_m"]
        )
        pix = cfg["grid_cell_native_px"]
        grid = np.zeros(len(uv), np.int64)
        grid[vis] = np.floor(uv[vis, 1] / pix).astype(int) * (
            int(np.ceil(c["width"] / pix)) + 1
        ) + np.floor(uv[vis, 0] / pix).astype(int)
        for a in aa:
            tree = int(a["gpsID"])
            poly = np.array(a["uv"])
            native = [a.get("native_width", c["width"]), a.get("native_height", c["height"])]
            if native != [c["width"], c["height"]]:
                raise ValueError(
                    f"Annotation/camera pixel dimensions disagree: {pid} {tree} {native}"
                )
            inside = MplPath(poly).contains_points(uv) & vis
            polygon = Polygon(poly)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            base = {
                "gpsID": tree,
                "photo_id": pid,
                "station": c["station_id"],
                "season": c["season"],
                "stratum": inv.loc[tree, "stratum"],
                "field_Htot": inv.loc[tree, "Hsup"],
                "reference_points": int(inside.sum()),
                "outline_touches_image_edge": bool(
                    (poly[:, 0].min() < 3)
                    | (poly[:, 1].min() < 3)
                    | (poly[:, 0].max() > c["width"] - 3)
                    | (poly[:, 1].max() > c["height"] - 3)
                ),
            }
            qa.append(base)
            variants = [("all_projected", vis, inside), ("front_depth_sensitivity", front, inside)]
            for tol in cfg["boundary_tolerance_native_px"][1:]:
                inner = polygon.buffer(-tol)
                outer = polygon.buffer(tol)
                sure_in = contains_xy(inner, uv[:, 0], uv[:, 1])
                sure_out = ~contains_xy(outer, uv[:, 0], uv[:, 1])
                variants.append((f"boundary_ignore_{tol}px", vis & (sure_in | sure_out), sure_in))
            for name, y in lab.items():
                pred = y == tree
                for variant, valid, truth in variants:
                    r = counts(truth, pred, valid)
                    r.update(
                        base,
                        method=name,
                        metric_basis=variant,
                        status="scored"
                        if np.count_nonzero(truth & valid) >= 20
                        else "insufficient_support",
                    )
                    rows.append(r)
                truth_cells = np.unique(grid[inside])
                pred_cells = np.unique(grid[pred & vis])
                tp = len(np.intersect1d(truth_cells, pred_cells))
                fp = len(pred_cells) - tp
                fn = len(truth_cells) - tp
                rows.append(
                    dict(
                        base,
                        method=name,
                        metric_basis="occupied_cells_24px",
                        status="scored" if len(truth_cells) >= 5 else "insufficient_support",
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        precision=tp / (tp + fp) if tp + fp else np.nan,
                        recall=tp / (tp + fn) if tp + fn else np.nan,
                        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                        iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                    )
                )
        print("Evaluated", pid, len(aa), "crowns", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(dest / "tables/evaluation_per_tree.csv", index=False)
    pd.DataFrame(qa).to_csv(dest / "qa/outline_support.csv", index=False)
    summary = []
    for scope, q in [
        ("all", t),
        ("low_stand", t[t.stratum == "low_stand"]),
        ("high_stand", t[t.stratum == "high_stand"]),
        ("focus_29", t[t.gpsID.isin(cfg["focus_ids"])]),
        ("pilot_10", t[t.gpsID.isin(cfg["pilot_ids"])]),
    ]:
        for (method, basis), g in q[q.status == "scored"].groupby(["method", "metric_basis"]):
            tp, fp, fn = g[["tp", "fp", "fn"]].sum()
            summary.append(
                dict(
                    scope=scope,
                    method=method,
                    metric_basis=basis,
                    n=len(g),
                    macro_f1=g.f1.mean(),
                    median_f1=g.f1.median(),
                    macro_iou=g.iou.mean(),
                    micro_f1=2 * tp / (2 * tp + fp + fn),
                    precision=g.precision.mean(),
                    recall=g.recall.mean(),
                )
            )
    s = pd.DataFrame(summary)
    s.to_csv(dest / "tables/evaluation_summary.csv", index=False)
    t[t.gpsID.isin(cfg["focus_ids"])].to_csv(dest / "tables/evaluation_focus_29.csv", index=False)
    base = t[(t.metric_basis == "all_projected") & (t.status == "scored")]
    base.groupby(["station", "method"]).f1.agg(["mean", "count"]).reset_index().to_csv(
        dest / "tables/evaluation_per_station.csv", index=False
    )
    ranking = s[(s.scope == "all") & (s.metric_basis == "all_projected")].sort_values(
        "macro_f1", ascending=False
    )
    winner = ranking.iloc[0].method
    # Paired station resampling describes variation of this reviewed dataset. Not an external validation CI.
    pivot = base.pivot(index=["station", "gpsID"], columns="method", values="f1")
    stations = pivot.index.get_level_values(0).unique()
    rng = np.random.default_rng(cfg["random_seed"])
    pairs = []
    for m in pivot.columns:
        if m == winner:
            continue
        dif = (pivot[winner] - pivot[m]).groupby(level=0).mean()
        b = np.array(
            [
                dif.reindex(rng.choice(stations, len(stations), replace=True)).mean()
                for _ in range(1000)
            ]
        )
        pairs.append(
            dict(
                best_observed=winner,
                comparison=m,
                paired_tree_mean_difference=float((pivot[winner] - pivot[m]).mean()),
                paired_station_mean_difference=float(dif.mean()),
                stations=len(stations),
                station_bootstrap_low=float(np.quantile(b, 0.025)),
                station_bootstrap_high=float(np.quantile(b, 0.975)),
                interpretation="equal-station mean resampling; paired tree mean is a different weighting; retrospective selection optimism remains",
            )
        )
    pd.DataFrame(pairs).to_csv(dest / "tables/method_comparisons.csv", index=False)
    write_json(
        dest / "qa/evaluation.json",
        {
            "best_observed_projection_agreement": winner,
            "ranking": ranking.to_dict("records"),
            "review_inputs": review_receipt,
            "cloud_sha256": sha(ROOT / "data/working/clouds_full.npz"),
            "label_sha256": sha(ROOT / "data/working/hybrid_labels.npz"),
            "reference_is_2d": True,
            "independent_test_set": False,
            "absolute_camera_accuracy_validated": False,
            "points_outside_reference_preserved_as_false_positives": True,
            "depth_filter_is_sensitivity_not_visibility_truth": True,
            "no_3d_precision_or_recall_claim": True,
        },
    )
    print(ranking[["method", "macro_f1", "macro_iou"]].to_string(index=False), flush=True)
    return t
