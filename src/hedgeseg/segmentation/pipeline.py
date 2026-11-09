"""Run all eleven assignments and write the label files.

Called by ``python run.py segment``. Takes about one minute on a laptop.

Inputs  : data/working/clouds_full.npz, inventory.csv (from ``run.py prepare``)
Outputs : data/working/six_method_labels_full.npz   six single methods
          data/working/hybrid_labels.npz            wider-window parents + combined
          data/working/baseline_labels.npz          the two comparators
          results/segmentation/*.csv, results/qa/segmentation.json
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .. import io
from . import constants as C
from .baselines import canopy_watershed, nearest_stem
from .canopy_peaks import add_peak_seeds, canopy_surface_and_basins, peak_seeds
from .combine import combine_by_margin
from .height import height_classes, height_penalty, selective_requests
from .methods import METHODS
from .scoring import pick_lowest_cost
from .seeds import add_wide_window_seeds, basal_seeds
from .stem_paths import stem_path_scores
from .transfer import transfer_to_uav


class Hedge:
    """Everything the methods need, loaded once."""

    def __init__(self):
        d = io.clouds()
        inv = io.inventory()
        self.ids = inv.gpsID.to_numpy()
        self.stems_xy = inv[["x", "y"]].to_numpy()
        self.is_lower = inv.stratum.eq("low_stand").to_numpy()
        self.field_height = inv.Hsup.to_numpy()
        self.upper_rows = np.flatnonzero(~self.is_lower)
        self.als_xyz, self.als_h = d["als_xyz"], d["als_hag"]
        self.uav_xyz, self.uav_h, self.uav_local = d["uav_xyz"], d["uav_hag"], d["uav_local"]
        self.frame_basis, self.frame_origin = d["frame_basis"], d["frame_origin"]
        nearest = cKDTree(d["ground_xy"]).query(self.stems_xy, k=C.GROUND_NEIGHBOURS)[1]
        self.ground_z = np.median(d["ground_z"][nearest], axis=1)
        self.stems_along = ((self.stems_xy - self.frame_origin) @ self.frame_basis.T)[:, 0]

    def caps(self, heights: str, requested=None):
        """Per-tree height cap (NaN = unconstrained) for a heights option."""
        if heights == "none":
            return np.full(len(self.ids), np.nan)
        if heights == "lower":
            return np.where(self.is_lower, self.field_height, np.nan)
        if heights == "classes":
            return np.where(self.is_lower, height_classes(self.field_height), np.nan)
        if heights == "requested":
            return np.where(requested, self.field_height, np.nan)
        raise ValueError(heights)

    def assign(self, als_scores, cap, allowance):
        """ALS labels from scores + height penalty, then transfer to UAV."""
        als = pick_lowest_cost(
            als_scores + height_penalty(self.als_xyz, self.ground_z, cap, allowance), self.ids
        )
        uav = transfer_to_uav(
            self.als_xyz,
            als["label"],
            self.uav_xyz,
            self.ids,
            self.stems_xy,
            self.ground_z,
            cap,
            allowance,
        )
        return als, uav

    def graphs(self, seeds):
        """Stem-path scores from basal seeds, and from basal + canopy-peak seeds."""
        basal_scores, _, diagnostics = stem_path_scores(
            self.als_xyz, self.als_h, self.stems_xy, self.ground_z, seeds=seeds
        )
        canopy = canopy_surface_and_basins(self.uav_local, self.uav_h)
        upper, records = peak_seeds(
            self.als_xyz,
            self.als_h,
            self.stems_xy,
            self.upper_rows,
            canopy,
            self.frame_basis,
            self.frame_origin,
        )
        peak_scores, _, _ = stem_path_scores(
            self.als_xyz,
            self.als_h,
            self.stems_xy,
            self.ground_z,
            seeds=add_peak_seeds(seeds, upper),
        )
        for r in records:
            r["tree"] = int(self.ids[r.pop("tree_row")])
        return basal_scores, peak_scores, diagnostics, records


def screen_wide_window_trees(hedge, used_wide, parents):
    """Reject recovered trees whose crown rises > 1.5 m above field height.

    A taller seed window can catch a taller neighbour's lower crown instead of
    the tree itself. That shows as an assigned top (99th percentile, >= 20
    points; stem-path parent first) far above the measured height; such a
    tree's labels are removed from all parents. The graph is not rerun.
    """
    g_uav = parents[1]
    t_uav = parents[3]
    rejected = []
    for j in np.flatnonzero(used_wide):
        tree = hedge.ids[j]
        own = g_uav["label"] == tree
        if own.sum() < C.FALLBACK_MIN_POINTS:
            own = t_uav["label"] == tree
        if own.sum() < C.FALLBACK_MIN_POINTS:
            continue
        top = float(np.quantile(hedge.uav_xyz[own, 2] - hedge.ground_z[j], C.FALLBACK_TOP_QUANTILE))
        if top > hedge.field_height[j] + C.FALLBACK_MAX_EXCESS:
            rejected.append(
                {
                    "tree": int(tree),
                    "top_m": round(top, 2),
                    "field_height_m": float(hedge.field_height[j]),
                }
            )
            for result in parents:
                result["label"][result["label"] == tree] = 0
            used_wide[j] = False
    return rejected


def run() -> None:
    """Stage ``segment``: compute all eleven assignments and write the label files."""
    t0 = time.time()
    out = io.RESULTS / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    hedge = Hedge()
    log = lambda msg: print(f"  {msg} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    print(
        f"{len(hedge.ids)} trees ({hedge.is_lower.sum()} lower / {(~hedge.is_lower).sum()} upper)"
        f" | ALS {len(hedge.als_xyz):,} | UAV {len(hedge.uav_xyz):,}",
        flush=True,
    )

    # ---- strict seeds: the six single methods --------------------------------
    strict = basal_seeds(hedge.als_xyz, hedge.als_h, hedge.stems_xy)
    basal_scores, peak_scores, graph_info, peak_records = hedge.graphs(strict)
    log(
        f"strict seeds for {sum(len(s) > 0 for s in strict)} trees; "
        f"{sum(r['used'] for r in peak_records)} upper trees carry a peak seed"
    )

    # trees whose height would be requested in the selective variant
    peak_only = pick_lowest_cost(peak_scores, hedge.ids)
    unconstrained = transfer_to_uav(
        hedge.als_xyz,
        peak_only["label"],
        hedge.uav_xyz,
        hedge.ids,
        hedge.stems_xy,
        hedge.ground_z,
        hedge.caps("none"),
        1.0,
    )
    requested = selective_requests(
        unconstrained["label"],
        hedge.uav_xyz,
        hedge.uav_local[:, 0],
        hedge.stems_along,
        hedge.ground_z,
        hedge.ids,
        hedge.is_lower,
    )
    log(f"{int(requested.sum())} height requests")

    six, summary, usage = {}, [], []
    scores_for = {"basal": basal_scores, "peaks": peak_scores}
    for m in METHODS:
        if m.graph is None or m.wide_window:
            continue
        cap = hedge.caps(m.heights, requested)
        als, uav = hedge.assign(scores_for[m.graph], cap, m.allowance)
        for sensor, result in (("als", als), ("uav", uav)):
            six[f"{m.code}_{sensor}_label"] = result["label"]
            six[f"{m.code}_{sensor}_candidate"] = result["label"]  # kept for file compatibility
            six[f"{m.code}_{sensor}_margin"] = result["margin"]
        summary.append(
            {
                "method": m.code,
                "name": m.name,
                "als_assigned": int((als["label"] > 0).sum()),
                "uav_assigned": int((uav["label"] > 0).sum()),
                "trees_with_uav": int(len(np.unique(uav["label"][uav["label"] > 0]))),
                "field_heights_used": int(np.isfinite(cap).sum()),
            }
        )
        usage += [
            {"method": m.code, "tree": int(t), "height_used": bool(np.isfinite(c)), "height_m": c}
            for t, c in zip(hedge.ids, cap)
        ]
        log(
            f"{m.code:<14} UAV {summary[-1]['uav_assigned']:,} points, "
            f"{summary[-1]['trees_with_uav']} trees"
        )

    # ---- wider seed window: the two parents and their combination ------------
    wide, used_wide = add_wide_window_seeds(strict, hedge.als_xyz, hedge.als_h, hedge.stems_xy)
    log("wider window recovers trees " + ", ".join(str(int(t)) for t in hedge.ids[used_wide]))
    basal_w, peak_w, _, _ = hedge.graphs(wide)
    g_als, g_uav = hedge.assign(basal_w, hedge.caps("lower"), 0.5)
    t_als, t_uav = hedge.assign(peak_w, hedge.caps("classes"), 0.5)
    rejected = screen_wide_window_trees(hedge, used_wide, [g_als, g_uav, t_als, t_uav])
    if rejected:
        log(f"wide-window seeds withdrawn: {rejected}")
    hybrid_als, _ = combine_by_margin(
        g_als["label"], t_als["label"], g_als["margin"], t_als["margin"]
    )
    hybrid_uav, took_path = combine_by_margin(
        g_uav["label"], t_uav["label"], g_uav["margin"], t_uav["margin"]
    )
    disagree = g_uav["label"] != t_uav["label"]
    log(f"parents disagree on {int(disagree.sum()):,} UAV points ({disagree.mean():.1%})")

    six.update(
        HYBRID_als_label=hybrid_als, HYBRID_uav_label=hybrid_uav, HYBRID_used_fallback=used_wide
    )
    np.savez_compressed(io.WORKING / "six_method_labels_full.npz", **six)
    np.savez_compressed(
        io.WORKING / "hybrid_labels.npz",
        hybrid_als_label=hybrid_als,
        hybrid_uav_label=hybrid_uav,
        hybrid_uav_margin=np.where(took_path, g_uav["margin"], t_uav["margin"]),
        G_uav_label=g_uav["label"],
        T_uav_label=t_uav["label"],
        G_als_label=g_als["label"],
        T_als_label=t_als["label"],
        G_uav_margin=g_uav["margin"],
        T_uav_margin=t_uav["margin"],
        used_fallback=used_wide,
        ids=hedge.ids,
    )

    # ---- comparators ---------------------------------------------------------
    nearest = nearest_stem(hedge.uav_xyz, hedge.stems_xy, hedge.ids)
    watershed, matches = canopy_watershed(
        hedge.uav_local,
        hedge.uav_h,
        hedge.stems_xy,
        hedge.ids,
        hedge.frame_basis,
        hedge.frame_origin,
    )
    np.savez_compressed(
        io.WORKING / "baseline_labels.npz", Nearest_stem_XY=nearest, Watershed=watershed
    )

    # ---- receipts --------------------------------------------------------------
    pd.DataFrame(summary).to_csv(out / "seg_full_summary.csv", index=False)
    pd.DataFrame(usage).to_csv(out / "seg_full_inventory_usage.csv", index=False)
    pd.DataFrame(peak_records).to_csv(out / "seg_full_apex_associations.csv", index=False)
    io.write_json(
        io.RESULTS / "qa" / "segmentation.json",
        {
            "trees": len(hedge.ids),
            "als_points": len(hedge.als_xyz),
            "uav_points": len(hedge.uav_xyz),
            "strict_seeds": int(sum(len(s) > 0 for s in strict)),
            "wide_window_trees": [int(t) for t in hedge.ids[used_wide]],
            "wide_window_rejected": rejected,
            "height_requests": int(requested.sum()),
            "peak_seeds_supported": int(sum(r["used"] for r in peak_records)),
            "graph": graph_info,
            "parents_disagree_uav": int(disagree.sum()),
            "watershed_matches": len(matches),
            "outline_used_in_assignment": False,
            "labels_sha256": io.sha(io.WORKING / "hybrid_labels.npz"),
            "seconds": round(time.time() - t0, 1),
        },
    )
    print(
        f"Done in {time.time() - t0:.0f} s; trees holding UAV points: "
        f"{len(np.unique(hybrid_uav[hybrid_uav > 0]))} of {len(hedge.ids)}",
        flush=True,
    )
