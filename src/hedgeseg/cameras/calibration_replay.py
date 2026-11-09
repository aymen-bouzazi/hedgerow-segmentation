"""Replay historical July rotation fits without replacing supplied cameras.

The fitting equations, 8 px robust scale, leave-one-out refits and free-focal
sensitivity follow the original FIT_july_all.py camera fit. This is
relative seasonal calibration, not an absolute cloud registration procedure.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..io import ROOT, cameras, project, rotation, sha, write_json

NAMES = ["ennezat_seasonal_review_3.json", "ennezat_seasonal_review_GB_station-31-34.json"]
CORE = ["station_id", "source_photo_id", "target_photo_id", "source_uv", "target_uv"]


def merge_ties(existing=()):
    """Keep original observations first; reject conflicting copies of a UID."""
    seen = {t["uid"]: dict(t) for t in existing}
    sources = []
    for name in NAMES:
        path = ROOT / "data/source/calibration_observations" / name
        if not path.exists():
            continue
        review = json.loads(path.read_text())
        for tie in review.get("tie_points", []):
            if tie["uid"] in seen:
                if any(seen[tie["uid"]].get(k) != tie.get(k) for k in CORE):
                    raise ValueError("Conflicting historical feature observation: " + tie["uid"])
            else:
                seen[tie["uid"]] = dict(tie)
        sources.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "sha256": sha(path),
                "tie_points": len(review.get("tie_points", [])),
            }
        )
    return list(seen.values()), sources


def fit(rays, target_uv, april, july, free_focal=False):
    """Fit one July camera orientation to April->July tie points (robust, 8 px scale)."""
    initial = np.r_[april["angles"], 0.0] if free_focal else np.asarray(april["angles"], float)

    def prediction(value, selected):
        camera = dict(
            july,
            camera=[0.0, 0.0, 0.0],
            angles=value[:3],
            focal_px=april["focal_px"] * np.exp(value[3]) if free_focal else april["focal_px"],
        )
        return project(selected, camera)[0]

    def residual(rs, uv):
        def objective(value):
            r = ((prediction(value, rs) - uv) / 8.0).ravel()
            return np.r_[r, value[3] / 0.02] if free_focal else r

        return objective

    opt = least_squares(
        residual(rays, target_uv), initial, loss="soft_l1", f_scale=1.0, max_nfev=800
    )
    error = np.linalg.norm(prediction(opt.x, rays) - target_uv, axis=1)
    loo = []
    for i in range(len(rays)):
        keep = np.arange(len(rays)) != i
        withheld = least_squares(
            residual(rays[keep], target_uv[keep]), opt.x, loss="soft_l1", f_scale=1.0, max_nfev=800
        )
        loo.append(float(np.linalg.norm(prediction(withheld.x, rays[i : i + 1])[0] - target_uv[i])))
    return opt, error, np.asarray(loo), prediction(opt.x, rays)


def run():
    """Stage ``camera-replay``: refit the July cameras from the saved tie points and compare with the supplied ones."""
    ties, sources = merge_ties()
    if len(sources) != len(NAMES):
        raise ValueError("Both historical July review exports are required for a complete replay.")
    ties = [
        t
        for t in ties
        if t["source_photo_id"].startswith("april/") and t["target_photo_id"].startswith("july/")
    ]
    cams = cameras()
    rows = []
    point_rows = []
    fitted = []
    for sid in sorted({t["station_id"] for t in ties}):
        obs = [t for t in ties if t["station_id"] == sid]
        source_ids = {t["source_photo_id"] for t in obs}
        target_ids = {t["target_photo_id"] for t in obs}
        if len(source_ids) != 1 or len(target_ids) != 1:
            raise ValueError("More than one photo pair at historical station " + str(sid))
        april = cams[next(iter(source_ids))]
        july = cams[next(iter(target_ids))]
        a = np.array([t["source_uv"] for t in obs])
        b = np.array([t["target_uv"] for t in obs])
        if len(obs) < 4 or len(np.unique(a, axis=0)) != len(a):
            raise ValueError("Too few or duplicate observations at station " + str(sid))
        rays = np.c_[
            (a - [april["width"] / 2, april["height"] / 2]) / april["focal_px"], np.ones(len(a))
        ] @ rotation(april["angles"])
        opt, error, loo, prediction = fit(rays, b, april, july)
        focal, ef, lf, _ = fit(rays, b, april, july, free_focal=True)
        supplied, _ = project(rays, dict(july, camera=[0.0, 0.0, 0.0]))
        rows.append(
            dict(
                station_id=sid,
                n_pairs=len(obs),
                fit_median_px=np.median(error),
                fit_max_px=error.max(),
                loo_median_px=np.median(loo),
                loo_max_px=loo.max(),
                focal_model_fit_median_px=np.median(ef),
                focal_model_loo_median_px=np.median(lf),
                focal_ratio=np.exp(focal.x[3]),
                refit_vs_supplied_median_px=np.median(
                    np.linalg.norm(prediction - supplied, axis=1)
                ),
            )
        )
        for i, t in enumerate(obs):
            point_rows.append(
                dict(uid=t["uid"], station_id=sid, fit_error_px=error[i], loo_error_px=loo[i])
            )
        fitted.append(
            dict(
                july,
                angles=opt.x[:3].tolist(),
                camera=april["camera"],
                focal_px=april["focal_px"],
                model="historical orientation replay; common April optical centre assumed",
            )
        )
    target = ROOT / "results/calibration"
    target.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(target / "july_replay_per_station.csv", index=False)
    pd.DataFrame(point_rows).to_csv(target / "july_replay_per_feature.csv", index=False)
    write_json(target / "july_replayed_cameras_NOT_APPLIED.json", fitted)
    receipt = {
        "historical_sources": sources,
        "unique_correspondences": len(ties),
        "stations_replayed": len(rows),
        "median_station_fit_px": float(table.fit_median_px.median()),
        "median_station_loo_px": float(table.loo_median_px.median()),
        "maximum_station_refit_vs_supplied_median_px": float(
            table.refit_vs_supplied_median_px.max()
        ),
        "model": "orientation only; April centre and focal fixed; 8 px robust residual scale",
        "source_cameras_overwritten": False,
        "absolute_camera_accuracy_validated": False,
    }
    write_json(ROOT / "results/qa/july_calibration_replay.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
