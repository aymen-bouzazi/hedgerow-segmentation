"""Geometric diagnostics of assigned returns and fitted envelopes.

An alpha-filtered Delaunay volume depends on sampling and its radius threshold.
Neither it nor occupied-cell volume measures living foliage. Ratios are
sensitivity descriptors, without a calibrated LAD correction or a known bound.
Occupancy and envelope containment use ALL assigned points; only Delaunay is
capped, with the cap exposed and tested in a separate paired analysis.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay

from ..io import ROOT, clouds, config, labels, write_json

ALPHA = 1.0  # m, circumradius ceiling for a kept tetrahedron
ALPHA_SWEEP = (0.6, 0.8, 1.0, 1.3, 1.8)
MIN_POINTS = 40  # below this a volume is not attempted
VOXEL = 0.25  # m, for the occupancy cross-check
MAX_POINTS = 4000  # Delaunay cost grows fast; subsample deterministically

# COORDINATES. Every geometric routine here runs on STEM-RELATIVE, GROUND-
# RELATIVE metres, never on Lambert-93. A circumradius is a difference of
# squared coordinates, so at easting 718,000 the squares are ~5e11 and the
# cancellation destroys the result: the same tree gives 71 m3 in local
# coordinates and 0.4 m3 in absolute ones. Delaunay itself degrades the same
# way. If this module is ever ported, keep the local frame.


def _tet_volumes(points, simplices):
    """Volume of each tetrahedron of a Delaunay triangulation."""
    a, b, c, d = (points[simplices[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _circumradii(points, simplices):
    """Circumradius of each tetrahedron.

    The circumcentre c of a tetrahedron with vertices a,b,c,d satisfies
    2(v - a).(x - a) = |v - a|^2 for each other vertex v, which is a 3x3 linear
    system per tetrahedron. Degenerate (coplanar) tetrahedra have a singular
    matrix; they are given an infinite radius so the alpha filter drops them.
    """
    p = points[simplices]
    rel = p[:, 1:] - p[:, 0][:, None, :]  # (n, 3, 3)
    sq = np.einsum("ijk,ijk->ij", rel, rel)  # (n, 3)
    out = np.full(len(simplices), np.inf)
    det = np.linalg.det(2 * rel)
    ok = np.isfinite(det) & (np.abs(det) > 1e-12)
    if ok.any():
        sol = np.linalg.solve((2 * rel)[ok], sq[ok][..., None])[..., 0]
        out[ok] = np.linalg.norm(sol, axis=1)
    return out


def alpha_volume(points, alpha=ALPHA):
    """Volume of the alpha shape. Returns (alpha_volume, convex_volume).

    `points` must already be in a LOCAL frame (see the note at the top of this
    module). Passing absolute projected coordinates silently returns nonsense.
    """
    points = np.asarray(points, float)
    if len(points) < 4:
        return np.nan, np.nan
    if np.abs(points).max() > 1e4:
        raise ValueError(
            "alpha_volume received coordinates larger than 10 km from the "
            "origin. Convert to a stem-relative, ground-relative frame first; "
            "the circumradius is numerically meaningless otherwise."
        )
    try:
        hull = ConvexHull(points)
        convex = float(hull.volume)
    except Exception:
        return np.nan, np.nan
    try:
        tri = Delaunay(points)
    except Exception:
        return np.nan, convex
    if not len(tri.simplices):
        return np.nan, convex
    vol = _tet_volumes(points, tri.simplices)
    rad = _circumradii(points, tri.simplices)
    keep = np.isfinite(rad) & (rad <= alpha) & np.isfinite(vol)
    return float(vol[keep].sum()), convex


def envelope_volume(rn, re, rs, rw, hcb, hlce, htot):
    """Two half-ellipsoids sharing the cross-section at HLCE.

    The horizontal section at HLCE is four quarter-ellipses, so its area is
    (pi/4)(rN.rE + rE.rS + rS.rW + rW.rN). Each half-ellipsoid contributes
    two thirds of that area times its own height.
    """
    if not np.all(np.isfinite([rn, re, rs, rw, hcb, hlce, htot])):
        return np.nan
    if not hcb <= hlce <= htot:
        return np.nan
    area = (np.pi / 4) * (rn * re + re * rs + rs * rw + rw * rn)
    return float((2 / 3) * area * (hlce - hcb) + (2 / 3) * area * (htot - hlce))


def inside_envelope(xyz, stem, ground_z, rn, re, rs, rw, hcb, hlce, htot, east, north):
    """Boolean mask: which of a tree's points lie inside its own envelope.

    Offsets are taken in the TRUE cardinal frame, because the radii were
    measured there. east and north are unit vectors in grid coordinates.
    """
    if not np.all(np.isfinite([rn, re, rs, rw, hcb, hlce, htot])) or htot <= hcb:
        return np.zeros(len(xyz), bool)
    d = xyz[:, :2] - np.asarray(stem, float)
    de, dn = d @ east, d @ north
    h = xyz[:, 2] - ground_z
    rx = np.where(de >= 0, re, rw)
    ry = np.where(dn >= 0, rn, rs)
    band = (h >= hcb) & (h <= htot)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = hlce - hcb
        upper = htot - hlce
        f = np.where(
            h <= hlce,
            np.sqrt(np.clip(1 - ((h - hlce) / max(lower, 1e-6)) ** 2, 0, None)),
            np.sqrt(np.clip(1 - ((h - hlce) / max(upper, 1e-6)) ** 2, 0, None)),
        )
        q = (de / np.maximum(rx * f, 1e-9)) ** 2 + (dn / np.maximum(ry * f, 1e-9)) ** 2
    return band & np.isfinite(q) & (q <= 1.0)


def triangulated_volumes(points, alphas):
    """Alpha-shape volumes at several radii plus the convex-hull volume."""
    if len(points) < 4:
        return [np.nan] * len(alphas), np.nan
    if np.abs(points).max() > 1e4:
        raise ValueError("Volume routines require local coordinates")
    try:
        convex = float(ConvexHull(points).volume)
    except Exception:
        return [np.nan] * len(alphas), np.nan
    try:
        tri = Delaunay(points)
        v = _tet_volumes(points, tri.simplices)
        r = _circumradii(points, tri.simplices)
    except Exception:
        return [np.nan] * len(alphas), convex
    return [float(v[np.isfinite(r) & (r <= a) & np.isfinite(v)].sum()) for a in alphas], convex


def alpha_volumes(points, thresholds):
    """Build one local Delaunay complex, then evaluate all requested radii."""
    points = np.asarray(points, float)
    if len(points) < 4:
        return {a: np.nan for a in thresholds}, np.nan
    if not np.isfinite(points).all() or np.abs(points).max() > 1e4:
        raise ValueError("Finite local coordinates within 10 km are required")
    try:
        convex = float(ConvexHull(points).volume)
    except Exception:
        return {a: np.nan for a in thresholds}, np.nan
    try:
        tri = Delaunay(points)
        vol = _tet_volumes(points, tri.simplices)
        rad = _circumradii(points, tri.simplices)
        valid = np.isfinite(rad) & np.isfinite(vol)
        return {a: float(vol[valid & (rad <= a)].sum()) for a in thresholds}, convex
    except Exception:
        return {a: np.nan for a in thresholds}, convex


def run():
    """Stage ``volumes``: envelope volume, alpha-shape sweep, containment and occupancy diagnostics."""
    from .parameters import support_mask

    cfg, d = config(), clouds()
    cell = cfg["occupancy_voxel_m"]
    alpha = cfg["alpha_radius_m"]
    cap = cfg["alpha_max_points"]
    uav = d["uav_xyz"]
    east, north = d["east"], d["north"]
    params = pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
    inv = pd.read_csv(ROOT / "data/working/inventory.csv").set_index("gpsID")
    L = labels()
    rows = []
    alpha_rows = []
    cap_rows = []
    for method, lab in sorted(L.items()):
        for r in params[params.method == method].itertuples():
            g = int(r.gpsID)
            pts = uav[lab == g]
            assigned = len(pts)
            if len(pts):
                pts = pts[support_mask(pts)]
            n = len(pts)
            rec = dict(
                method=method,
                gpsID=g,
                n_points=assigned,
                n_supported=n,
                stratum=r.stratum,
                geometry_population="supported assigned points, same filter as fitted profiles; no outline clipping",
            )
            if n < MIN_POINTS:
                rows.append(
                    dict(
                        rec,
                        n_delaunay_points=0,
                        n_occupied_voxels=0,
                        alpha_volume_m3=np.nan,
                        convex_volume_m3=np.nan,
                        envelope_volume_m3=np.nan,
                        envelope_over_alpha=np.nan,
                        shell_enclosed=np.nan,
                        occupied_voxel_volume_m3=np.nan,
                        note="too few assigned points for volume diagnostic",
                    )
                )
                continue
            stem = inv.loc[g, ["x", "y"]].to_numpy(float)
            gz = float(inv.at[g, "ground_z"])
            rel = pts[:, :2] - stem
            local = np.c_[rel @ east, rel @ north, pts[:, 2] - gz]
            # These are full-data descriptors: the Delaunay cap cannot change them.
            nvox = len(np.unique(np.floor(local / cell).astype(np.int64), axis=0))
            shell = (
                float(
                    inside_envelope(
                        pts,
                        stem,
                        gz,
                        r.rnorth,
                        r.reast,
                        r.rsouth,
                        r.rwest,
                        r.hcb,
                        r.hlce,
                        r.htot,
                        east,
                        north,
                    ).mean()
                )
                if np.isfinite(r.hlce)
                else np.nan
            )
            ev = envelope_volume(r.rnorth, r.reast, r.rsouth, r.rwest, r.hcb, r.hlce, r.htot)
            sample = local[np.random.default_rng(g).permutation(n)[:cap]] if n > cap else local
            thresholds = (
                sorted(set(cfg["alpha_radius_sweep_m"] + [alpha]))
                if method == "Hybrid"
                else [alpha]
            )
            vols, cv = alpha_volumes(sample, thresholds)
            av = vols[alpha]
            ratio = ev / av if np.isfinite(ev) and np.isfinite(av) and av > 0 else np.nan
            rows.append(
                dict(
                    rec,
                    n_delaunay_points=len(sample),
                    delaunay_capped=n > cap,
                    n_occupied_voxels=nvox,
                    alpha_volume_m3=av,
                    convex_volume_m3=cv,
                    envelope_volume_m3=ev,
                    envelope_over_alpha=ratio,
                    shell_enclosed=shell,
                    occupied_voxel_volume_m3=nvox * cell**3,
                    note="",
                )
            )
            if method == "Hybrid":
                for a, v in vols.items():
                    alpha_rows.append(
                        dict(
                            gpsID=g,
                            alpha_m=a,
                            n_delaunay_points=len(sample),
                            alpha_volume_m3=v,
                            envelope_volume_m3=ev,
                            envelope_over_alpha=(
                                ev / v if np.isfinite(ev) and np.isfinite(v) and v > 0 else np.nan
                            ),
                        )
                    )
                if n > cap and np.isfinite(ev):
                    # Every capped Hybrid fit: repeated subsets and full assigned population.
                    for target in (1000, 2000, cap, n):
                        for replicate in range(1 if target == n else 2):
                            size = min(n, target)
                            sub = (
                                local
                                if size == n
                                else local[
                                    np.random.default_rng(g + 100000 * replicate).permutation(n)[
                                        :size
                                    ]
                                ]
                            )
                            vv, _ = alpha_volumes(sub, [alpha])
                            v = vv[alpha]
                            cap_rows.append(
                                dict(
                                    gpsID=g,
                                    assigned_points=n,
                                    point_limit=target,
                                    sampled_points=len(sub),
                                    replicate=replicate,
                                    alpha_m=alpha,
                                    alpha_volume_m3=v,
                                    envelope_volume_m3=ev,
                                    envelope_over_alpha=ev / v
                                    if np.isfinite(v) and v > 0
                                    else np.nan,
                                    full_occupied_voxels=nvox,
                                    sampled_occupied_voxels=len(
                                        np.unique(np.floor(sub / cell).astype(np.int64), axis=0)
                                    ),
                                )
                            )
        print("Geometric diagnostics", method, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/crown_volumes.csv", index=False)
    aa = pd.DataFrame(alpha_rows)
    aa.to_csv(ROOT / "results/tables/crown_volume_alpha_per_tree.csv", index=False)
    valid_ids = (
        aa.pivot(index="gpsID", columns="alpha_m", values="envelope_over_alpha").dropna().index
    )
    sweep = []
    for a, q in aa.groupby("alpha_m"):
        ratios = q.envelope_over_alpha.dropna()
        paired = q[q.gpsID.isin(valid_ids)].envelope_over_alpha
        sweep.append(
            dict(
                alpha_m=a,
                n_trees=len(ratios),
                median_envelope_over_alpha=ratios.median(),
                q25=ratios.quantile(0.25),
                q75=ratios.quantile(0.75),
                paired_trees=len(paired),
                paired_median_envelope_over_alpha=paired.median(),
            )
        )
    pd.DataFrame(sweep).to_csv(ROOT / "results/tables/crown_volume_alpha_sweep.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(
        ROOT / "results/tables/volume_point_count_sensitivity.csv", index=False
    )
    h = out[(out.method == "Hybrid") & out.envelope_over_alpha.notna()]
    write_json(
        ROOT / "results/qa/volumes.json",
        dict(
            alpha_m=alpha,
            voxel_m=cell,
            delaunay_point_cap=cap,
            trees_with_volume=len(h),
            capped_Hybrid_trees=int(
                out[(out.method == "Hybrid")].delaunay_capped.fillna(False).sum()
            ),
            point_count_sensitivity_trees=len({r["gpsID"] for r in cap_rows}),
            median_envelope_over_alpha=float(h.envelope_over_alpha.median()),
            median_shell_enclosed=float(h.shell_enclosed.median()),
            alpha_sweep_median_range=[
                float(min(s["median_envelope_over_alpha"] for s in sweep)),
                float(max(s["median_envelope_over_alpha"] for s in sweep)),
            ],
            occupancy_before_point_cap=True,
            support_matches_parameter_fit=True,
            containment_uses_full_supported_points=True,
            interpretation="geometric sensitivity of sampled returns and fitted envelopes; not foliage volume or LAD bias",
            known_bound_on_real_crown_volume=False,
            independent_volume_validation=False,
            grid_frame="stem-relative true east/north and stem-local ground; per-tree origin differs from global evaluation grid",
        ),
    )
    print("Hybrid geometric ratio:", float(h.envelope_over_alpha.median()), flush=True)
