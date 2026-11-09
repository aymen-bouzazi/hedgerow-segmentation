"""Occupancy-weighted agreement with photograph outlines, not 3D ground truth.

Both sides use actual in-frame point projections and the same fixed voxel IDs.
The primary domain is the full image. A separate spatial-window sensitivity
records exclusions; it never replaces the primary result or clips predictions
by the reference polygon. Empty cells are unknown, not a filled crown interior.
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from ..io import clouds, config, inventory, labels, write_json
from .context import projected_frames, reference_inputs, result_root


def run(review_path=None, output_dir=None):
    """Stage ``occupancy``: agreement counted in occupied 0.25 m voxels instead of points (removes density weighting)."""
    cfg = config()
    d = clouds()
    inv = inventory().set_index("gpsID")
    L = labels()
    _, cams, aa, receipt = reference_inputs(review_path)
    dest = result_root(output_dir)
    xyz = d["uav_xyz"]
    sd = d["uav_plot_sd"]
    origin = np.asarray(cfg["hedge_origin"])
    basis = np.asarray(cfg["hedge_basis"])
    frames = list(projected_frames(xyz, cams, aa))
    rows = []
    omissions = []
    primary = cfg.get("occupancy_voxel_m", 0.25)
    for cell in sorted(set([primary] + cfg.get("occupancy_voxel_sweep_m", [0.2, 0.25, 0.5]))):
        # A single origin for every method/reference; world vertical displacement,
        # not HAG projected back through an invented voxel-centre terrain height.
        key = np.floor(np.c_[sd, xyz[:, 2] - xyz[:, 2].min()] / cell).astype(np.int64)
        _, voxel = np.unique(key, axis=0, return_inverse=True)
        for pid, c, uv, depth, domain, outlines in frames:
            for a in outlines:
                g = a["gpsID"]
                inside = domain & MplPath(np.asarray(a["uv"], float)).contains_points(uv)
                if inside.sum() < 20:
                    continue
                s0 = ((inv.loc[g, ["x", "y"]].to_numpy(float) - origin) @ basis.T)[0]
                window = (abs(sd[:, 0] - s0) <= cfg.get("diagnostic_window_half_length_m", 9.0)) & (
                    abs(sd[:, 1]) <= cfg.get("diagnostic_slab_half_width_m", 3.0)
                )
                for scope, mask in [
                    ("full_image", domain),
                    ("window_slab_sensitivity", domain & window),
                ]:
                    ref = set(np.unique(voxel[inside & mask]).tolist())
                    for method, lab in L.items():
                        predmask = (lab == g) & mask
                        pred = set(np.unique(voxel[predmask]).tolist())
                        tp = len(ref & pred)
                        fp = len(pred - ref)
                        fn = len(ref - pred)
                        rows.append(
                            dict(
                                gpsID=g,
                                photo_id=pid,
                                station=c["station_id"],
                                stratum=inv.at[g, "stratum"],
                                method=method,
                                domain=scope,
                                voxel_m=cell,
                                reference_points=int((inside & mask).sum()),
                                prediction_points=int(predmask.sum()),
                                reference_voxels=len(ref),
                                prediction_voxels=len(pred),
                                tp_voxels=tp,
                                fp_voxels=fp,
                                fn_voxels=fn,
                                iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                                recall=tp / (tp + fn) if tp + fn else np.nan,
                                precision=tp / (tp + fp) if tp + fp else np.nan,
                                reference_occupied_cell_volume_proxy_m3=len(ref) * cell**3,
                            )
                        )
                        if cell == primary and scope == "full_image":
                            p = (lab == g) & domain
                            omissions.append(
                                dict(
                                    gpsID=g,
                                    method=method,
                                    photo_id=pid,
                                    reference_points_omitted=int((inside & ~window).sum()),
                                    prediction_points_omitted=int((p & ~window).sum()),
                                    false_positive_points_omitted=int(
                                        (p & ~window & ~inside).sum()
                                    ),
                                )
                            )
    x = pd.DataFrame(rows)
    x.to_csv(dest / "tables/occupancy_outline_per_tree.csv", index=False)
    pd.DataFrame(omissions).to_csv(dest / "tables/occupancy_window_omissions.csv", index=False)
    summary = []
    for (method, scope, cell), group in x.groupby(["method", "domain", "voxel_m"]):
        summary.append(
            dict(
                method=method,
                domain=scope,
                voxel_m=cell,
                n=group.iou.notna().sum(),
                mean_iou=group.iou.mean(),
                median_iou=group.iou.median(),
                mean_recall=group.recall.mean(),
                mean_precision=group.precision.mean(),
                **{
                    "mean_iou_" + s: group.loc[group.stratum == s, "iou"].mean()
                    for s in ["low_stand", "high_stand"]
                },
            )
        )
    S = pd.DataFrame(summary)
    S.to_csv(dest / "tables/occupancy_outline_summary.csv", index=False)
    main = S[(S.domain == "full_image") & (S.voxel_m == primary)].sort_values(
        "mean_iou", ascending=False
    )
    main.to_csv(dest / "tables/carve3d_summary.csv", index=False)
    # Compatibility view, explicitly named domain. No legacy centre projections.
    q = x[(x.domain == "full_image") & (x.voxel_m == primary)]
    wide = q.pivot(
        index=["gpsID", "stratum"], columns="method", values=["iou", "recall", "precision"]
    )
    wide.columns = [b + "_" + a for a, b in wide.columns]
    wide.reset_index().to_csv(dest / "tables/carve3d_per_tree.csv", index=False)
    write_json(
        dest / "qa/carve3d.json",
        dict(
            voxel_m=primary,
            voxel_sweep_m=sorted(x.voxel_m.unique()),
            trees_scored=int(q.gpsID.nunique()),
            method_mean_iou=dict(zip(main.method, main.mean_iou)),
            domain="full native image rectangle and positive depth",
            spatial_window_is_secondary_only=True,
            point_projection_used_on_both_sides=True,
            reference_clipping_of_predictions=False,
            reference="same photographic outlines and supplied cloud support; occupancy-weighted agreement, not independent 3D truth",
            empty_cells="unknown; occupied-cell volume is neither filled crown volume nor a calibrated bound",
            review_inputs=receipt,
        ),
    )
    print(main.to_string(index=False), flush=True)
