"""Rebuild the original 3D inspection tool from current segmentation labels.

Alpha surfaces and shortest-path lines are visual diagnostics, not measured
biological crown volumes or reconstructed branches. Form fitting uses observed
directional profiles, never the photographic evaluation polygons.
"""

import json

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, QhullError, cKDTree

from ..io import (
    ROOT,
    annotations,
    cameras,
    clouds,
    inventory,
    labels,
    project,
    write_json,
)
from ..structure.parameters import DIRECTIONS
from . import TEMPLATES

FORMS = [
    "half_ellipsoids",
    "ellipsoid",
    "cone_up",
    "cone_down",
    "paraboloid_up",
    "paraboloid_down",
    "box",
]


def alpha_surface(points, radius=1.0):
    """Display surface of a tree's points (alpha shape, visual only)."""
    if len(points) < 5:
        return None
    # Real representatives per 0.25 m cube; only for display surface construction.
    _, ii = np.unique(np.floor(points / 0.25).astype(np.int32), axis=0, return_index=True)
    p = points[np.sort(ii)]
    if len(p) < 5:
        return None
    try:
        tet = Delaunay(p).simplices
    except QhullError:
        return None
    q = p[tet]
    a = q[:, 1:] - q[:, :1]
    valid = abs(np.linalg.det(a)) > 1e-9
    t = tet[valid]
    a = a[valid]
    b = (a * a).sum(2) / 2
    if not len(t):
        return None
    c = np.linalg.solve(a, b[..., None])[..., 0]
    t = t[(c * c).sum(1) <= radius**2]
    if not len(t):
        return None
    faces = np.concatenate([t[:, k] for k in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]])
    faces, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    faces = faces[count == 1]
    vertices, idx = np.unique(faces, return_inverse=True)
    return {"v": np.round(p[vertices], 3).ravel().tolist(), "f": idx.tolist()}


def support_paths(p):
    """Shortest-path lines through a tree's ALS points (visual only, not branches)."""
    if len(p) < 3:
        return []
    pairs = cKDTree(p).query_pairs(0.9, output_type="ndarray")
    if not len(pairs):
        return []
    distance = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
    w = distance * (1 + (distance / 0.6) ** 2)
    graph = coo_matrix(
        (np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
        shape=(len(p), len(p)),
    ).tocsr()
    root = np.argmin(np.linalg.norm(p[:, :2], axis=1) + 0.05 * np.maximum(p[:, 2], 0))
    _, parent = dijkstra(graph, directed=False, indices=int(root), return_predecessors=True)
    ix = np.flatnonzero(parent >= 0)
    return np.round(np.stack([p[ix], p[parent[ix]]], axis=1), 3).ravel().tolist()


def factor(name, f, h=0.5):
    """Width profile of the alternative crown shapes drawn in the 3D viewer."""
    f = np.clip(f, 0, 1)
    if name == "box":
        return np.ones_like(f)
    if name == "ellipsoid":
        return np.sqrt(np.maximum(0, 1 - (2 * f - 1) ** 2))
    if name == "half_ellipsoids":
        return np.sqrt(np.maximum(0, 1 - ((f - h) / np.where(f < h, h, 1 - h)) ** 2))
    if name == "cone_up":
        return 1 - f
    if name == "cone_down":
        return f
    if name == "paraboloid_up":
        return np.sqrt(1 - f)
    return np.sqrt(f)


def fitted_forms(pr, row):
    """Fit alternative crown shapes to a tree's profile for display."""
    if not np.isfinite(row.get("htot", np.nan)) or not np.isfinite(row.get("hlce", np.nan)):
        return {}
    hcb, htot = float(row.hcb), float(row.htot)
    pr = pr[(pr.height > hcb) & (pr.height < htot)]
    if len(pr) < 4:
        return {}
    yy = pr[DIRECTIONS].to_numpy()
    valid = np.isfinite(yy)
    if valid.sum(0).min() < 3:
        return {}
    y = np.array(
        [
            median_filter(
                pd.Series(yy[:, k]).interpolate(limit_direction="both").to_numpy(),
                size=3,
                mode="nearest",
            )
            for k in range(4)
        ]
    ).T
    f = (pr.height.to_numpy() - hcb) / (htot - hcb)
    out = {}
    for name in FORMS:
        if name == "half_ellipsoids":
            r = np.array([row["r" + n] for n in DIRECTIONS])
            h = (float(row.hlce) - hcb) / (htot - hcb)
        else:
            h = 0.5 if name in ["box", "ellipsoid"] else 0 if name.endswith("_up") else 1
            k = factor(name, f, h)
            fit = least_squares(
                lambda r: (k[:, None] * r - y)[valid],
                np.maximum(np.nanmax(y, axis=0), 0.1),
                bounds=(np.full(4, 0.05), np.maximum(np.nanmax(y, axis=0) * 3, 0.2)),
                loss="soft_l1",
                f_scale=0.25,
            )
            r = fit.x
        err = (factor(name, f, h)[:, None] * r - y)[valid]
        out[name] = {
            "rN": float(r[0]),
            "rE": float(r[1]),
            "rS": float(r[2]),
            "rW": float(r[3]),
            "hcb": hcb,
            "htot": htot,
            "hlce": hcb + h * (htot - hcb),
            "fh": h,
            "profile_rmse_m": float(np.sqrt(np.mean(err**2))),
            "name": name,
        }
    return out


def run():
    """Build inspect_trees_3D.html (3D trees, sections, fitted crowns)."""
    d = clouds()
    inv = inventory()
    labs = labels()
    cams = cameras()
    params = pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
    profiles = pd.read_csv(ROOT / "results/parameters/vertical_profiles.csv")
    pp = {(m, int(g)): q for (m, g), q in profiles.groupby(["method", "gpsID"])}
    six = dict(np.load(ROOT / "data/working/six_method_labels_full.npz"))
    hy = dict(np.load(ROOT / "data/working/hybrid_labels.npz"))
    B = d["frame_basis"]
    origin = d["frame_origin"]
    z0 = float(inv.ground_z.median())
    scene = {
        "methods": {},
        "reference_indices": {},
        "frame_basis": B.tolist(),
        "origin": origin.tolist(),
        "ground_origin": z0,
        "surface_radius_m": 1.0,
        "surface_voxel_m": 0.25,
    }
    forms_rows = []
    for m, y in labs.items():
        al = six.get(m + "_als_label")
        if m in ["Hybrid", "G_fallback", "T_fallback"]:
            al = hy.get(
                {
                    "Hybrid": "hybrid_als_label",
                    "G_fallback": "G_als_label",
                    "T_fallback": "T_als_label",
                }[m]
            )
        entries = {}
        for t in inv.itertuples():
            ix = np.flatnonzero(y == t.gpsID)
            p = d["uav_xyz"][ix] - [t.x, t.y, t.ground_z]
            p[:, :2] = p[:, :2] @ B.T
            ai = np.flatnonzero(al == t.gpsID) if al is not None else np.array([], int)
            a = d["als_xyz"][ai] - [t.x, t.y, t.ground_z]
            a[:, :2] = a[:, :2] @ B.T
            q = params[(params.method == m) & (params.gpsID == t.gpsID)].iloc[0]
            forms = fitted_forms(
                pp.get((m, t.gpsID), pd.DataFrame(columns=["height", *DIRECTIONS])), q
            )
            entries[str(t.gpsID)] = {
                "alpha": alpha_surface(p),
                "paths": support_paths(a),
                "als_indices": ai.tolist(),
                "forms": forms,
            }
            for name, v in forms.items():
                forms_rows.append(
                    dict(
                        gpsID=t.gpsID,
                        method=m,
                        form=name,
                        **{
                            k: v[k]
                            for k in [
                                "hcb",
                                "htot",
                                "hlce",
                                "rN",
                                "rE",
                                "rS",
                                "rW",
                                "profile_rmse_m",
                            ]
                        },
                    )
                )
        scene["methods"][m] = entries
        print("3D geometry:", m, flush=True)
    ann = annotations()
    primary = ann.get("primary_reference_uids")
    aa = [
        a
        for a in ann["annotations"]
        if a.get("type") == "visible_crown"
        and (not primary or primary.get(str(a["gpsID"])) == a["uid"])
    ]
    for pid in {a["photo_id"] for a in aa}:
        c = cams[pid]
        uv, z = project(d["uav_xyz"], c)
        valid = (
            (z > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < c["width"])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < c["height"])
        )
        valid &= abs(d["uav_plot_sd"][:, 1]) <= 3
        for a in [a for a in aa if a["photo_id"] == pid]:
            scene["reference_indices"][str(a["gpsID"])] = np.flatnonzero(
                valid & MplPath(a["uv"]).contains_points(uv)
            ).tolist()
    pd.DataFrame(forms_rows).to_csv(
        ROOT / "results/parameters/visual_form_comparisons.csv", index=False
    )
    (ROOT / "viewers/assets/shapes.js").write_text(
        "window.ENNEZAT_SHAPES=" + json.dumps(scene, separators=(",", ":"), allow_nan=False) + ";",
        encoding="utf-8",
    )
    html = (TEMPLATES / "original_ennezat_trees_3D.html").read_text()
    html = (
        html[: html.index("<script src=")]
        + '<script src="assets/three.min.js"></script><script src="assets/data.js"></script><script src="assets/shapes.js"></script><script>'
        + (TEMPLATES / "trees_restored.js").read_text()
        + "</script></body></html>"
    )
    html = html.replace(
        "Ennezat crowns &mdash; G_lower_050", "Ennezat · tree and section inspection"
    )
    html = html.replace(
        '<label>Tree</label><select id="tree"></select>',
        '<label>Segmentation method</label><select id="method"></select><label>Individual tree</label><select id="tree"></select><label><input id="useRange" type="checkbox"> Show an ID interval</label><div><input id="fromID" type="number" min="1" max="194" value="1" style="width:75px"> to <input id="toID" type="number" min="1" max="194" value="5" style="width:75px"><button id="applyRange">Show</button></div>',
    )
    html = (
        html.replace(
            "Grown tree &mdash; branches generated into the measured crown",
            "ALS graph support · not biological branches",
        )
        .replace("show the tree", "shortest paths")
        .replace(">foliage<", ">UAV support<")
    )
    html = (
        html.replace(
            "Reconstructed solid &mdash; sections stacked along the hedge",
            "Occupied space · 0.25 m cells",
        )
        .replace("from your delineation", "outline-consistent space")
        .replace("from the method", "assigned space")
    )
    html = html.replace(
        "Carving check &mdash; your outline against the method",
        "Single-view consistency · depth remains ambiguous",
    )
    html = html.replace("grown tree", "ALS graph paths").replace(">foliage<br>", ">UAV support<br>")
    helptext = """<details class="hint"><summary>What do these layers mean?</summary><p>Points are the selected cloud assigned by the chosen method. Neighbours provide spatial context. Alpha shape is a display envelope (1 m circumradius; 0.25 m real-point sampling), not measured foliage volume. Occupancy shows cubes containing returns; empty cubes are not evidence of empty crowns.</p><p>ALS paths follow connections at most 0.9 m long from a near-stem root within the assigned ALS points. They illustrate support connectivity, not recovered branches. Disconnected components remain disconnected.</p><p>Outline-consistent space uses occupied points inside one photographic outline and a ±3 m hedge-depth slab. It is not 3D ground truth. Green/purple/red layers show agreement, assignment outside this support and support not assigned, respectively.</p><p>Each envelope is fitted to directional height profiles with field HCB held fixed. Alternative forms are comparisons, not validated tree reconstructions. Four radii use true cardinal directions and are rotated into hedge coordinates for display. Along view looks along the hedge; across view shows its long side.</p><p>Drag to rotate, right-drag to pan, wheel to zoom. ID interval is inclusive. Slice position is along the hedge relative to the selected group centre.</p></details>"""
    html = html.replace('<table id="info">', helptext + '<table id="info">')
    (ROOT / "viewers/inspect_trees_3D.html").write_text(html, encoding="utf-8")
    write_json(
        ROOT / "results/qa/restored_3d_build.json",
        {
            "methods": len(labs),
            "inventory": len(inv),
            "all_shapes_from_current_labels": True,
            "reference_space_is_not_ground_truth": True,
            "forms_fitted_without_photo_outlines": True,
        },
    )
