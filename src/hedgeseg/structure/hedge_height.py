"""Segmentation-independent surface summaries over a fixed longitudinal extent.

Upper-profile, crest and occupied-raster means answer different questions.
Empty bins/cells remain unknown. All results inherit supplied sampling, ground
normalisation, corridor selection and minimum HAG; no wind-protection distance
or absolute canopy accuracy is inferred.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter1d

from ..io import ROOT, clouds, config, write_json


def envelope_profile(s, h, step=0.25, extent=None):
    """Highest point in each 0.25 m interval along the hedge (the upper silhouette)."""
    lo, hi = extent if extent is not None else (s.min(), s.max())
    n = max(1, int(np.ceil((hi - lo) / step)))
    i = np.floor((s - lo) / step).astype(int)
    i = np.clip(i, 0, n - 1)
    p = np.full(n, -np.inf)
    np.maximum.at(p, i, h)
    p[np.isneginf(p)] = np.nan
    return lo + (np.arange(n) + 0.5) * step, p


def crests(x, prof, separation, step=0.25, smooth_m=0.75):
    """Local maxima of the smoothed upper profile at least ``separation`` metres apart."""
    valid = np.isfinite(prof)
    size = max(1, int(round(smooth_m / step)))
    numer = uniform_filter1d(np.where(valid, prof, 0), size, mode="nearest")
    denom = uniform_filter1d(valid.astype(float), size, mode="nearest")
    sm = np.divide(numer, denom, out=np.full_like(numer, -np.inf), where=denom > 0)
    w = max(1, int(np.ceil(separation / step)))
    w += (w + 1) % 2
    cand = np.flatnonzero(valid & (prof >= 1.5) & (sm == maximum_filter(sm, w, mode="nearest")))
    keep = []
    for i in sorted(cand, key=lambda i: (-sm[i], x[i])):
        if all(abs(x[i] - x[j]) >= separation for j in keep):
            keep.append(i)
    return np.asarray(sorted(keep), int)


def run():
    """Stage ``hedge-height``: upper-profile mean, crest mean and area-based mean height of the hedge."""
    cfg = config()
    d = clouds()
    sd = d["uav_plot_sd"]
    uh = d["uav_hag"].astype(float)
    step = cfg.get("canopy_step_m", 0.25)
    cor = cfg.get("canopy_corridor_half_width_m", 3.0)
    sep = cfg.get("canopy_crest_min_separation_m", 2.0)
    base = (abs(sd[:, 1]) <= cor) & np.isfinite(uh)
    extent = (float(sd[base, 0].min()), float(sd[base, 0].max()))
    length = extent[1] - extent[0]
    inlength = (sd[:, 0] >= extent[0]) & (sd[:, 0] <= extent[1]) & np.isfinite(uh)

    def make(c, s):
        mask = inlength & (abs(sd[:, 1]) <= c)
        return envelope_profile(sd[mask, 0], uh[mask], s, extent)

    x, prof = make(cor, step)
    cov = np.isfinite(prof)
    H = float(np.nanmean(prof))
    pd.DataFrame(dict(along_hedge_m=x, surface_height_m=prof, observed=cov)).to_csv(
        ROOT / "results/tables/canopy_profile.csv", index=False
    )
    rows = []
    for c, s, kind in [(cor, s, "step") for s in [0.1, 0.25, 0.5, 1.0, 2.0]] + [
        (c, step, "corridor") for c in [1.5, 2.0, 3.0, 5.0, 8.0]
    ]:
        _, p = make(c, s)
        v = p[np.isfinite(p)]
        rows.append(
            dict(
                quantity="observed_upper_profile_mean",
                sweep=kind,
                step_m=s,
                corridor_m=c,
                n=len(v),
                total_bins=len(p),
                coverage=float(len(v) / len(p)),
                H_m=float(v.mean()),
                sd_m=float(v.std(ddof=1)),
                min_m=float(v.min()),
                max_m=float(v.max()),
                extent_start_m=extent[0],
                extent_end_m=extent[1],
            )
        )
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/canopy_envelope_H.csv", index=False)
    cs = []
    cp = []
    for separation in sorted(set([sep, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])):
        k = crests(x, prof, separation, step)
        v = prof[k]
        cs.append(
            dict(
                min_separation_m=separation,
                n_crests=len(v),
                crest_mean_m=float(v.mean()) if len(v) else np.nan,
                crest_sd_m=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                crests_per_100m=100 * len(v) / length,
            )
        )
        for i in k:
            cp.append(dict(min_separation_m=separation, along_hedge_m=x[i], height_m=prof[i]))
    pd.DataFrame(cs).to_csv(ROOT / "results/tables/canopy_crests.csv", index=False)
    pd.DataFrame(cp).to_csv(ROOT / "results/tables/canopy_crest_points.csv", index=False)
    ar = []
    for cell in [0.2, 0.25, 0.5, 1.0]:
        s = sd[base, 0]
        h = uh[base]
        di = sd[base, 1]
        si = np.floor((s - extent[0]) / cell).astype(int)
        dj = np.floor((di + cor) / cell).astype(int)
        shape = (int(np.ceil(length / cell)), int(np.ceil(2 * cor / cell)))
        si = np.clip(si, 0, shape[0] - 1)
        dj = np.clip(dj, 0, shape[1] - 1)
        g = np.full(shape, -np.inf)
        np.maximum.at(g, (si, dj), h)
        v = g[np.isfinite(g)]
        for threshold in [None, 2.0]:
            a = v if threshold is None else v[v > threshold]
            ar.append(
                dict(
                    cell_m=cell,
                    min_height_threshold_m=np.nan if threshold is None else threshold,
                    n_cells=len(a),
                    total_rectangle_cells=int(g.size),
                    observed_cell_fraction=len(v) / g.size,
                    area_m2=len(a) * cell**2,
                    area_weighted_mean_m=float(a.mean()),
                    domain="occupied cells only; empty footprint cells unknown",
                )
            )
    pd.DataFrame(ar).to_csv(ROOT / "results/tables/canopy_area_weighted.csv", index=False)
    chosen = next(r for r in cs if r["min_separation_m"] == sep)
    write_json(
        ROOT / "results/qa/canopy.json",
        dict(
            corridor_half_width_m=cor,
            step_m=step,
            fixed_longitudinal_extent_m=list(extent),
            hedge_length_m=length,
            profile_steps=len(prof),
            steps_with_canopy=int(cov.sum()),
            coverage=float(cov.mean()),
            envelope_H_m=H,
            crest_separation_chosen_m=sep,
            crest_separation_basis="declared 2 m surface-feature scale; not selected from inventory stem count",
            crest_mean_m=chosen["crest_mean_m"],
            n_crests=chosen["n_crests"],
            upstream_minimum_hag_m=cfg["minimum_hag_m"],
            interpretation="upper-profile mean, crest mean and occupied-raster mean are scale-dependent surface descriptors. No inferred wind-protection range. Empty cells are not assigned zero height.",
        ),
    )
    print(
        f"Upper-profile mean {H:.3f} m; {chosen['n_crests']} crests at {sep:g} m separation, mean {chosen['crest_mean_m']:.3f} m; observed coverage {cov.mean():.1%}",
        flush=True,
    )
