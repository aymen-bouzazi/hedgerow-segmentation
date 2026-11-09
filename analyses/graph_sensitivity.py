"""How much do the two graph constants matter?   python analyses/graph_sensitivity.py

Varies the maximum link length (0.5-1.5 m) and the link-cost length scale
(0.3-1.5 m, or no penalty) for the stem-path method without heights
(G_none), keeping seeds and transfer fixed, and scores each run against the
photo outlines. Also reports how connected the ALS graph is at each length.

Writes results/tables/graph_link_sensitivity.csv. Retrospective: scored on
the same outlines used to develop the method, so it identifies settings to
revisit, not validated optima. Takes ~10 minutes.
"""

import time

import numpy as np
import pandas as pd
from _common import PhotoScorer, io
from hedgeseg.segmentation import constants as C
from hedgeseg.segmentation import stem_paths
from hedgeseg.segmentation.pipeline import Hedge
from hedgeseg.segmentation.scoring import pick_lowest_cost
from hedgeseg.segmentation.seeds import basal_seeds
from hedgeseg.segmentation.transfer import transfer_to_uav


def run_once(hedge, seeds, max_link, cost_scale):
    original = C.LINK_COST_SCALE
    if cost_scale is None:  # pure link length, no gap penalty
        stem_paths.link_cost = lambda length: length
    else:
        C.LINK_COST_SCALE = cost_scale
    try:
        scores, _, info = stem_paths.stem_path_scores(
            hedge.als_xyz,
            hedge.als_h,
            hedge.stems_xy,
            hedge.ground_z,
            seeds=seeds,
            max_link=max_link,
        )
    finally:
        C.LINK_COST_SCALE = original
        stem_paths.link_cost = lambda length: length * (1 + (length / C.LINK_COST_SCALE) ** 2)
    als = pick_lowest_cost(scores, hedge.ids)
    uav = transfer_to_uav(
        hedge.als_xyz,
        als["label"],
        hedge.uav_xyz,
        hedge.ids,
        hedge.stems_xy,
        hedge.ground_z,
        hedge.caps("none"),
        1.0,
    )
    return uav["label"], info


def main():
    hedge = Hedge()
    scorer = PhotoScorer(hedge.uav_xyz)
    seeds = basal_seeds(hedge.als_xyz, hedge.als_h, hedge.stems_xy)
    trials = [(cap, 0.6) for cap in (0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)]
    trials += [(0.9, s) for s in (0.3, 0.45, 0.9, 1.5, None)]
    rows = []
    for cap, scale in trials:
        t0 = time.time()
        labels, info = run_once(hedge, seeds, cap, scale)
        rows.append(
            {
                "max_link_m": cap,
                "cost_scale_m": scale if scale else "none",
                "largest_piece_fraction": info["largest_piece_fraction"],
                "reachable_fraction": info["reachable_fraction"],
                "trees_with_points": int(len(np.unique(labels[labels > 0]))),
                "mean_f1": scorer.mean_f1(labels),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(io.RESULTS / "tables" / "graph_link_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
