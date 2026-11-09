"""A prioritised review queue and a deliberately stratified profile atlas.

Review priority does not convert provisional fits into accepted biology. All
individuals remain in the table. Photographic performance and model support
are separate review axes, not automatic criteria for removing surveyed trees.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..io import ROOT, write_json
from .parameters import DIRECTIONS, crown_factor


def run():
    """Stage ``review-queue``: rank trees for manual review and draw a profile atlas."""
    p = (
        pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
        .query("method=='Hybrid'")
        .copy()
    )
    e = pd.read_csv(ROOT / "results/tables/evaluation_per_tree.csv").query(
        "method=='Hybrid' and metric_basis=='all_projected'"
    )[["gpsID", "f1", "iou"]]
    p = p.merge(e, on="gpsID", how="left")

    p["review_priority"] = np.select(
        [
            ~p.status.str.startswith("model_fit"),
            p.status.eq("model_fit_review"),
            p.allometry_requires_review,
        ],
        [1, 2, 3],
        default=4,
    )
    p["next_action"] = np.select(
        [
            ~p.status.str.startswith("model_fit"),
            p.status.eq("model_fit_review"),
            p.allometry_requires_review,
        ],
        [
            "Inspect ID attribution and directional support; retain missing geometry until measured or supported",
            "Inspect observed profiles and crown shape; a mathematical fit is provisional",
            "Resolve per-stem basal area before using circumference in allometry",
        ],
        default="Retain as internally supported; independent crown measurements still needed for biological validation",
    )
    p.sort_values(["review_priority", "f1", "gpsID"]).to_csv(
        ROOT / "results/tables/structural_review_queue.csv", index=False
    )
    profiles = pd.read_csv(ROOT / "results/parameters/vertical_profiles.csv").query(
        "method=='Hybrid'"
    )
    cases = []
    for status in sorted(p.status.unique()):
        for stratum in ["low_stand", "high_stand"]:
            g = p[(p.status == status) & (p.stratum == stratum)].sort_values(
                ["profile_rmse_m", "gpsID"], na_position="last"
            )
            if len(g):
                cases.extend([int(g.iloc[0].gpsID), int(g.iloc[-1].gpsID)])
    cases = list(dict.fromkeys(cases))
    if 165 not in cases:
        cases.append(165)
    for g in p.sort_values("profile_rmse_m", ascending=False).gpsID:
        if len(cases) >= 12:
            break
        if g not in cases:
            cases.append(int(g))
    cases = cases[:12]
    if 165 not in cases:
        cases[-1] = 165
    out = ROOT / "results/diagnostics"
    out.mkdir(exist_ok=True)
    palette = ["#236b8e", "#d68131", "#347c4d", "#9b4d80"]
    for start in range(0, len(cases), 4):
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
        for ax, g in zip(axes.flat, cases[start : start + 4]):
            row = p[p.gpsID == g].iloc[0]
            pr = profiles[profiles.gpsID == g]
            for k, n in enumerate(DIRECTIONS):
                if len(pr):
                    ax.plot(pr[n], pr.height, ".", color=palette[k], ms=3, label=n)
                if str(row.status).startswith("model_fit"):
                    z = np.linspace(row.hcb, row.htot, 150)
                    ax.plot(
                        crown_factor(z, row.hcb, row.hlce, row.htot) * row["r" + n],
                        z,
                        color=palette[k],
                        lw=1,
                    )
            if row.status.startswith("model_fit"):
                ax.axhline(row.hlce, color="grey", ls="--", lw=0.8)
            ax.axhline(row.hcb, color="black", ls=":", lw=0.8)
            ax.set(
                title=f"ID {g} · {row.stratum}\n{row.status}",
                xlabel="Directional radius (m)",
                ylabel="Height from stem ground (m)",
            )
            ax.grid(alpha=0.15)
            if len(pr):
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No supported directional profile",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
        for ax in list(axes.flat)[len(cases[start : start + 4]) :]:
            ax.set_visible(False)
        fig.savefig(out / f"profile_atlas_{start // 4 + 1}.png", dpi=180)
        fig.savefig(out / f"profile_atlas_{start // 4 + 1}.svg")
        plt.close(fig)
    write_json(
        ROOT / "results/qa/quality.json",
        dict(
            review_priority_counts={
                str(k): int(v) for k, v in p.review_priority.value_counts().items()
            },
            atlas_ids=cases,
            selection="first and last profile RMSE in each fit-status/stratum group; include ID165; fill remaining slots with high misfit. Illustrative coverage, not a blind validation sample.",
            review_does_not_change_labels_or_fit_acceptance=True,
        ),
    )
    print("Review queue for", len(p), "individuals; atlas IDs", cases, flush=True)
