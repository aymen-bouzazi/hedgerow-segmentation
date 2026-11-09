"""Retrospective comparison of parent-combination policies on the shared image domain.

This diagnostic does not select a new production policy. The production rule
preserves sole-parent labels, then compares cost gaps for two positive claims.
The historical tie rule is retained as a regression comparator.
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from ..io import ROOT, clouds, inventory, sha, write_json
from ..segmentation.combine import combine_by_margin as combine_parents
from .context import projected_frames, reference_inputs, result_root


def rules(gl, tl, gm, tm, low_ids):
    """Alternative ways to merge the two wider-window parents (e.g. lower-stratum parent wins)."""
    g_low, t_low = np.isin(gl, low_ids), np.isin(tl, low_ids)
    a = np.where(g_low, gl, 0)
    b = np.where(~t_low & (tl > 0), tl, 0)
    c = np.where(g_low, gl, np.where(~t_low & (tl > 0), tl, 0))
    current, _ = combine_parents(gl, tl, gm, tm)
    return {
        "G alone": gl.copy(),
        "T alone": tl.copy(),
        "low wins, conflicts dropped": np.where(a == 0, np.where(t_low, 0, tl), a),
        "low wins, nothing dropped": np.where(g_low, gl, np.where(tl > 0, tl, gl)),
        "high wins, conflicts dropped": np.where(b == 0, np.where(g_low, gl, 0), b),
        "each owns its stratum, gaps filled": np.where(c == 0, np.where(gl > 0, gl, tl), c),
        "larger margin wins": current,
        "legacy margin ties G": np.where(gm >= tm, gl, tl),
    }


def _scorer(review_path=None):
    """Prepare the projected outlines once so each merge rule can be scored quickly."""
    _, cameras, outlines, receipt = reference_inputs(review_path)
    refs, visible = [], {}
    inv = inventory().set_index("gpsID")
    for pid, camera, uv, depth, domain, aa in projected_frames(
        clouds()["uav_xyz"], cameras, outlines
    ):
        visible[pid] = domain
        for a in aa:
            inside = np.flatnonzero(
                domain & MplPath(np.asarray(a["uv"], float)).contains_points(uv)
            )
            if len(inside) >= 20:
                refs.append((a["gpsID"], pid, inside))

    def score(labels):
        n = int(inv.index.max()) + 1
        sizes = {
            pid: np.bincount(labels[mask].astype(int), minlength=n) for pid, mask in visible.items()
        }
        rows = []
        for g, pid, inside in refs:
            tp = int(np.count_nonzero(labels[inside] == g))
            pred = int(sizes[pid][g])
            fp = pred - tp
            fn = len(inside) - tp
            rows.append(
                dict(
                    gpsID=g,
                    photo_id=pid,
                    station=cameras[pid]["station_id"],
                    stratum=inv.at[g, "stratum"],
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                )
            )
        return pd.DataFrame(rows)

    return score, len(refs), receipt


def run(review_path=None, output_dir=None):
    """Stage ``conflict``: score the alternative merge rules against the outlines (diagnostic only)."""
    h = dict(np.load(ROOT / "data/working/hybrid_labels.npz"))
    gl, tl, gm, tm = (h[k] for k in ("G_uav_label", "T_uav_label", "G_uav_margin", "T_uav_margin"))
    inv = inventory()
    variants = rules(gl, tl, gm, tm, inv.loc[inv.stratum == "low_stand", "gpsID"].to_numpy())
    score, nrefs, review_receipt = _scorer(review_path)
    dest = result_root(output_dir)
    rows, per = [], []
    for name, labels in variants.items():
        s = score(labels)
        s["rule"] = name
        per.append(s)
        rows.append(
            dict(
                rule=name,
                macro_f1=float(s.f1.mean()),
                low=float(s[s.stratum == "low_stand"].f1.mean()),
                high=float(s[s.stratum == "high_stand"].f1.mean()),
                trees_with_points=int(len(np.unique(labels[labels > 0]))),
                labelled_points=int((labels > 0).sum()),
            )
        )
    x = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    x.to_csv(dest / "tables/conflict_rules.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(dest / "tables/conflict_per_tree.csv", index=False)
    disagree = gl != tl
    both = disagree & (gl > 0) & (tl > 0)
    g_low = np.isin(gl, inv.loc[inv.stratum == "low_stand", "gpsID"])
    t_low = np.isin(tl, inv.loc[inv.stratum == "low_stand", "gpsID"])
    production = h["hybrid_uav_label"]
    legacy = variants["legacy margin ties G"]
    write_json(
        dest / "qa/conflict.json",
        dict(
            reference_trees=nrefs,
            tested_rules=len(variants),
            uav_points=len(gl),
            agreement_fraction=float((~disagree).mean()),
            disagreements=int(disagree.sum()),
            contested_both_claim=int(both.sum()),
            contested_G_low_T_high=int((both & g_low & ~t_low).sum()),
            contested_G_high_T_low=int((both & ~g_low & t_low).sum()),
            ranking={r.rule: float(r.macro_f1) for r in x.itertuples()},
            best_rule=x.iloc[0].rule,
            best_macro_f1=float(x.iloc[0].macro_f1),
            production_labels_match_rule=[
                k for k, v in variants.items() if np.array_equal(v, production)
            ],
            parent_union_labelled_points=int(((gl > 0) | (tl > 0)).sum()),
            production_unlabelled_parent_claims=int(
                ((production == 0) & ((gl > 0) | (tl > 0))).sum()
            ),
            changed_from_legacy_rule=int((production != legacy).sum()),
            review_inputs=review_receipt,
            label_sha256=sha(ROOT / "data/working/hybrid_labels.npz"),
            selection_note="Descriptive ranking on reviewed references. Production preserves sole-parent coverage; no rule is selected automatically from these scores.",
            margin_note="uncalibrated cost separation, not a probability",
        ),
    )
    print(x.to_string(index=False), flush=True)
    return x
