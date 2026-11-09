"""Supported cloud envelopes and explicit field HCB -> constrained crown representation.

The fitted HLCE is a model parameter, not a claim that all four directions have
their observed maximum at the same height. Raw directional maxima remain saved.
"""

import re

import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..io import ROOT, clouds, config, inventory, labels, write_json

DIRECTIONS = ["north", "east", "south", "west"]


def support_mask(points):
    """Keep points with >= 3 neighbours within 0.6 m (removes isolated returns before measuring)."""
    cfg = config()
    return cKDTree(points).query_ball_point(
        points, cfg.get("parameter_support_radius_m", 0.6), return_length=True
    ) >= cfg.get("parameter_support_neighbors", 3)


def cardinal_offsets(xy, stem):
    """World projected axes, not hedge axes. Small grid convergence retained explicitly."""
    cfg = config()
    gamma = np.radians(
        Proj("EPSG:2154").get_factors(cfg["longitude"], cfg["latitude"]).meridian_convergence
    )
    # True north has grid-coordinate direction (-sin gamma, cos gamma).
    delta = xy - np.asarray(stem)
    north = np.array([-np.sin(gamma), np.cos(gamma)])
    east = np.array([np.cos(gamma), np.sin(gamma)])
    return np.c_[delta @ north, delta @ east, -delta @ north, -delta @ east]


def crown_factor(z, hcb, hlce, htot):
    """Relative width of the two-half-ellipsoid crown at height z (0 at HCB and top, 1 at HLCE)."""
    h = np.where(z >= hlce, htot - hlce, hlce - hcb)
    return np.sqrt(np.maximum(0, 1 - ((z - hlce) / np.maximum(h, 1e-8)) ** 2))


def profile(xyz, stem, ground, bin_m, q, min_points):
    """Directional crown extent per 0.5 m height band: 95th percentile of positive N/E/S/W offsets."""
    z = xyz[:, 2] - ground
    off = cardinal_offsets(xyz[:, :2], stem)
    rows = []
    if not len(z):
        return pd.DataFrame()
    for b in np.arange(0, max(z) + bin_m, bin_m):
        sel = (z >= b) & (z < b + bin_m)
        if sel.sum() < min_points:
            continue
        row = {"height": b + bin_m / 2, "points": int(sel.sum())}
        for k, name in enumerate(DIRECTIONS):
            v = off[sel, k]
            v = v[v > 0]
            row[name] = np.quantile(v, q) if len(v) >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_profile(pr, hcb, htot):
    """Fit HLCE and four radii to the smoothed profiles (soft-L1, bounded) and scan 41 HLCE candidates."""
    pr = pr[(pr.height > hcb) & (pr.height < htot)].copy()
    z = pr.height.to_numpy()
    y = pr[DIRECTIONS].to_numpy()
    if len(z) < 4 or np.sum(np.isfinite(y), axis=0).min() < 3:
        return None
    smooth = np.array(
        [
            median_filter(
                pd.Series(y[:, k]).interpolate(limit_direction="both").to_numpy(),
                size=3,
                mode="nearest",
            )
            for k in range(4)
        ]
    ).T
    peak = float(z[np.argmax(np.nanmean(smooth, axis=1))])
    base = np.nanmax(smooth, axis=0)
    low = hcb + 0.05
    high = htot - 0.05
    if high <= low:
        return None
    start = np.r_[np.clip(peak, low + 0.001, high - 0.001), np.maximum(base, 0.1)]
    valid = np.isfinite(y)

    def residual(v):
        return (crown_factor(z, hcb, v[0], htot)[:, None] * v[1:] - smooth)[valid]

    bounds = (np.r_[low, np.full(4, 0.05)], np.r_[high, np.maximum(base * 2, 0.2)])
    # Every grid candidate uses exactly the fit's finite mask, smoothed response,
    # radius bounds and soft-L1 scale. Fixed-height radius problems are convex.
    # Vectorised bisection solves their monotone first derivatives, including
    # constrained minima at either radius bound.
    grid = np.linspace(low, high, config().get("hlce_grid_size", 41))
    f = crown_factor(z[None, :], hcb, grid[:, None], htot)[:, :, None]
    lower = np.full((len(grid), 4), 0.05)
    upper = np.broadcast_to(bounds[1][1:], lower.shape).copy()
    for _ in range(40):
        mid = (lower + upper) / 2
        res = f * mid[:, None, :] - smooth[None, :, :]
        derivative = np.sum(
            np.where(valid[None, :, :], f * res / np.sqrt(1 + (res / 0.25) ** 2), 0), axis=1
        )
        lower = np.where(derivative < 0, mid, lower)
        upper = np.where(derivative >= 0, mid, upper)
    radii = (lower + upper) / 2
    res = f * radii[:, None, :] - smooth[None, :, :]
    cost = (
        np.sum(
            np.where(valid[None, :, :], 2 * 0.25**2 * (np.sqrt(1 + (res / 0.25) ** 2) - 1), 0),
            axis=(1, 2),
        )
        / valid.sum()
    )
    best = int(np.argmin(cost))
    starts = [start, np.r_[grid[best], radii[best]]]
    fits = [least_squares(residual, s, bounds=bounds, loss="soft_l1", f_scale=0.25) for s in starts]
    good = [v for v in fits if v.success and np.isfinite(v.cost)]
    if not good:
        return None
    fit = min(good, key=lambda v: v.cost)
    fit_cost = 2 * fit.cost / valid.sum()
    grid = np.r_[grid, fit.x[0]]
    cost = np.r_[cost, fit_cost]
    near = grid[cost <= cost.min() + max(0.01, cost.min() * 0.10)]
    return {
        "hlce": fit.x[0],
        **{"r" + n: fit.x[k + 1] for k, n in enumerate(DIRECTIONS)},
        "profile_rmse_m": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
        "profile_soft_l1_mean_loss": float(fit_cost),
        "optimizer_success": bool(fit.success),
        "optimizer_status": int(fit.status),
        "hlce_profile_low": float(near.min()),
        "hlce_profile_high": float(near.max()),
        "hlce_near_boundary": bool(
            (fit.x[0] - hcb) < 0.08 * (htot - hcb) or (htot - fit.x[0]) < 0.08 * (htot - hcb)
        ),
        "n_profile_bins": len(pr),
        "directional_peak_heights": {
            n: float(z[np.argmax(smooth[:, k])]) for k, n in enumerate(DIRECTIONS)
        },
    }


def girth(t):
    """Interpret the field circumference record (single stem, listed stems or whole clump)."""
    raw = str(t.circumference_raw)
    nums = [float(v) for v in re.findall(r"\d+(?:[.,]\d+)?", raw.replace(",", "."))]
    mean = float(t.circ_cm_avg)
    n = float(t.trunk_avg)
    listed = t.circ_avg_basis == "individual_values_mean" and len(nums) >= 3
    if listed:
        ceq = float(np.linalg.norm(nums))
        lo = hi = ceq
        basis = "basal-area-equivalent from confirmed listed individual circumferences"
    elif n <= 1:
        ceq = mean
        lo = min(nums) if nums else mean
        hi = max(nums) if nums else mean
        basis = "single stem; midpoint when range"
    else:
        ceq = np.nan
        lo = hi = np.nan
        basis = "whole-clump circumference; not basal-area-equivalent"
    clump = n > 1 and not listed
    return {
        "gbh_cm": ceq,
        "gbh_low_cm": lo,
        "gbh_high_cm": hi,
        "dbh_cm": ceq / np.pi,
        "girth_basis": basis,
        "stem_count_used": n,
        "field_circumference_raw": raw,
        "field_mean_girth_cm": mean,
        "clump_circumference_cm": mean if clump else np.nan,
        "clump_circumference_low_cm": min(nums) if clump and nums else np.nan,
        "clump_circumference_high_cm": max(nums) if clump and nums else np.nan,
        "clump_circular_equivalent_diameter_cm": mean / np.pi if clump else np.nan,
        "dbh_interpretation": (
            "basal-area-equivalent of confirmed listed stems"
            if listed
            else (
                "single-stem perimeter-derived diameter"
                if n <= 1
                else "unresolved individual-stem DBH; clump perimeter cannot determine basal area"
            )
        ),
        "allometry_requires_review": bool(clump),
        "listed_girth_semantics_require_review": False,
    }


def run():
    """Stage ``parameters``: supported height, field HCB, crown envelope and review flags for every tree and method."""
    cfg = config()
    d = clouds()
    inv = inventory()
    labs = labels()
    rows = []
    profiles = []
    sensitivity = []
    for method, lab in labs.items():
        for t in inv.itertuples():
            points = d["uav_xyz"][lab == t.gpsID]
            flags = []
            row = {
                "gpsID": int(t.gpsID),
                "method": method,
                "stratum": t.stratum,
                "species_field": t.variety,
                "state": t.state,
                "x": t.x,
                "y": t.y,
                "ground_z": t.ground_z,
                "n_assigned": len(points),
                "Htot_field": t.Hsup,
                "HCB_field": t.Hinf,
                "HLCE_field": t.HLCE,
                "hcb": t.Hinf,
                "hcb_source": "field observation; zero means foliage reaches ground (user confirmed 7 October 2026)",
                "observed_3d_validation": False,
                **girth(t),
            }
            if len(points) < 20:
                rows.append(
                    dict(
                        row,
                        status="unresolved_insufficient_points",
                        qa_flags="No synthetic crown inserted",
                    )
                )
                continue
            # Remove isolated supports only; not points outside a hand-drawn reference.
            support = support_mask(points)
            pts = points[support]
            row["n_supported"] = len(pts)
            row["removed_isolated"] = len(points) - len(pts)
            if len(pts) < 20:
                rows.append(
                    dict(
                        row,
                        status="unresolved_insufficient_support",
                        qa_flags="No synthetic crown inserted",
                    )
                )
                continue
            heights = pts[:, 2] - t.ground_z
            top = float(np.quantile(heights, 0.995))
            row.update(
                htot=top,
                htot_supported_max=float(max(heights)),
                htot_q99=float(np.quantile(heights, 0.99)),
                htot_source="UAV supported 99.5th percentile, stem-local ALS ground",
            )
            if not np.isfinite(t.Hinf) or t.Hinf < 0 or t.Hinf >= top - 0.1:
                rows.append(
                    dict(
                        row,
                        status="unresolved_HCB_conflict",
                        qa_flags="Field HCB cannot form a crown below observed top; not reset to zero",
                    )
                )
                continue
            pr = profile(
                pts,
                [t.x, t.y],
                t.ground_z,
                cfg["parameter_bin_m"],
                cfg["parameter_quantile"],
                cfg["parameter_min_points_per_bin"],
            )
            if pr.empty:
                rows.append(
                    dict(row, status="unresolved_profile", qa_flags="No supported height bins")
                )
                continue
            for p in pr.to_dict("records"):
                profiles.append(dict(p, gpsID=t.gpsID, method=method))
            # Sensor-conditional observations remain usable even if a crown model fails.
            for k, name in enumerate(DIRECTIONS):
                row["observed_" + name + "_extent_q95_m"] = float(
                    np.quantile(np.maximum(cardinal_offsets(pts[:, :2], [t.x, t.y])[:, k], 0), 0.95)
                )
                finite = pr[name].dropna()
                row["observed_" + name + "_profile_peak_height_m"] = (
                    float(pr.loc[finite.idxmax(), "height"]) if len(finite) else np.nan
                )
            width = pr[DIRECTIONS].max(axis=1).to_numpy()
            thresh = max(0.3, 0.15 * np.nanmax(width))
            large = width >= thresh
            onset = [
                i
                for i in range(len(large) - 2)
                if large[i : i + 3].all()
                and np.all(np.diff(pr.height.to_numpy()[i : i + 3]) < cfg["parameter_bin_m"] * 1.1)
            ]
            row["hcb_width_proxy_not_exported"] = (
                float(pr.height.iloc[onset[0]] - cfg["parameter_bin_m"] / 2) if onset else np.nan
            )
            fit = fit_profile(pr, t.Hinf, top)
            if fit is None:
                rows.append(
                    dict(
                        row,
                        status="unresolved_directional_support",
                        qa_flags="Too few supported bins or directions; no mean-radius filling",
                    )
                )
                continue
            peaks = fit.pop("directional_peak_heights")
            row.update(**fit, **{"hlce_observed_" + n: peaks[n] for n in DIRECTIONS})
            for bm, q in [(0.35, 0.90), (0.5, 0.95), (0.75, 0.98)]:
                alt = fit_profile(profile(pts, [t.x, t.y], t.ground_z, bm, q, 8), t.Hinf, top)
                if alt:
                    sensitivity.append(
                        dict(
                            gpsID=t.gpsID,
                            method=method,
                            bin_m=bm,
                            quantile=q,
                            hlce=alt["hlce"],
                            **{"r" + n: alt["r" + n] for n in DIRECTIONS},
                        )
                    )
            row["crown_volume_m3"] = (
                np.pi
                / 6
                * (row["reast"] + row["rwest"])
                * (row["rnorth"] + row["rsouth"])
                * (top - t.Hinf)
            )
            row["crown_centroid_stem_offset_m"] = float(
                np.linalg.norm(np.median(pts[:, :2], axis=0) - [t.x, t.y])
            )
            if fit["hlce_near_boundary"]:
                flags.append("HLCE near model boundary")
            if fit["hlce_profile_high"] - fit["hlce_profile_low"] > 0.25 * (top - t.Hinf):
                flags.append("HLCE weakly identified by profile")
            if row["profile_rmse_m"] > 0.75:
                flags.append("Large crown-profile misfit")
            if t.Hinf >= t.HLCE:
                flags.append("Field HCB >= field HLCE; raw values retained")
            if t.state != "B":
                flags.append("Inventory vitality/state needs explicit simulation choice")
            row.update(
                status="model_fit_review" if flags else "model_fit",
                qa_flags="; ".join(flags),
                structural_use_status=(
                    "provisional_model_requires_review"
                    if flags
                    else "internally_supported_model; no independent biological validation"
                ),
                parameter_interpretation="shared-HLCE ellipsoidal approximation, not direct botanical measurements",
            )
            rows.append(row)
        print("Extracted profiles", method, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/parameters/parameters_all_methods.csv", index=False)
    pd.DataFrame(profiles).to_csv(ROOT / "results/parameters/vertical_profiles.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(
        ROOT / "results/parameters/profile_sensitivity.csv", index=False
    )
    table[table.gpsID.isin(cfg["pilot_ids"])].to_csv(
        ROOT / "results/parameters/pilot_10_all_methods.csv", index=False
    )
    table[table.gpsID.isin(cfg["focus_ids"])].to_csv(
        ROOT / "results/parameters/focus_29_all_methods.csv", index=False
    )
    write_json(
        ROOT / "results/qa/parameters.json",
        {
            "status_counts": table.groupby(["method", "status"])
            .size()
            .reset_index(name="n")
            .to_dict("records"),
            "field_HCB_used": True,
            "field_heights_used_in_some_segmentation_methods": True,
            "HLCE_is_model_fitted": True,
            "fit_and_height_grid_share_objective": True,
            "loss": "soft_l1; f_scale=0.25 m; same finite mask and radius bounds",
            "HLCE_grid_size": cfg.get("hlce_grid_size", 41),
            "near_optimal_band_rule": "mean loss <= minimum + max(0.01 m2, 0.1 * minimum); heuristic, not confidence interval",
            "radial_directions": "true cardinal approximation using Lambert-93 convergence at site centre",
            "profile_sensitivity_is_not_statistical_confidence_interval": True,
            "missing_parameters_not_silently_filled": True,
            "field_measurement_conventions": cfg["field_measurement_conventions"],
        },
    )
    from .summary import run as summarize_structure

    summarize_structure()
