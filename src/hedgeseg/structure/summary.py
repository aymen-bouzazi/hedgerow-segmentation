"""Descriptive structural summaries; no photographic trimming or new fitting."""

import numpy as np
import pandas as pd

from ..io import ROOT, write_json


def run():
    """Count fitted, flagged and unresolved crowns for the combined method."""
    p = pd.read_csv(ROOT / "results/parameters/parameters_all_methods.csv")
    statuses = p.groupby(["method", "status"]).size().reset_index(name="individuals")
    statuses.to_csv(ROOT / "results/tables/structural_status_counts.csv", index=False)
    rows = []
    quantities = [
        "htot",
        "hcb",
        "hlce",
        "rnorth",
        "reast",
        "rsouth",
        "rwest",
        "crown_volume_m3",
        "profile_rmse_m",
    ]
    for method, g in p.groupby("method"):
        for key in quantities:
            v = g[key].dropna()
            rows.append(
                dict(
                    method=method,
                    parameter=key,
                    n=len(v),
                    mean=v.mean(),
                    median=v.median(),
                    q10=v.quantile(0.1),
                    q90=v.quantile(0.9),
                    minimum=v.min(),
                    maximum=v.max(),
                    population="all nonmissing values; model radii/volume only exist for fitted crowns",
                )
            )
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/structural_distributions.csv", index=False)
    hy = p.query("method=='Hybrid'").set_index("gpsID")
    g = p.query("method=='G_fallback'").set_index("gpsID")
    pairs = []
    for key in ["htot", "hlce", "rnorth", "reast", "rsouth", "rwest", "crown_volume_m3"]:
        finite = np.isfinite(hy[key]) & np.isfinite(g[key])
        v = (hy.loc[finite, key] - g.loc[finite, key]).abs()
        pairs.append(
            dict(
                parameter=key,
                paired_individuals=len(v),
                median_absolute_difference=v.median(),
                q90_absolute_difference=v.quantile(0.9),
                maximum_absolute_difference=v.max(),
                interpretation="label-method sensitivity, not error against independent measurements",
            )
        )
    pd.DataFrame(pairs).to_csv(
        ROOT / "results/tables/structural_method_sensitivity.csv", index=False
    )
    sensitivity = pd.read_csv(ROOT / "results/parameters/profile_sensitivity.csv").query(
        "method=='Hybrid'"
    )
    spans = []
    for id, q in sensitivity.groupby("gpsID"):
        for key in ["hlce", "rnorth", "reast", "rsouth", "rwest"]:
            spans.append(
                dict(gpsID=id, parameter=key, recipes=len(q), range=q[key].max() - q[key].min())
            )
    pd.DataFrame(spans).to_csv(
        ROOT / "results/tables/Hybrid_profile_recipe_ranges.csv", index=False
    )
    flags = hy.qa_flags.fillna("")
    valid = hy.status.str.startswith("model_fit")
    receipt = {
        "Hybrid_inventory": len(hy),
        "Hybrid_fitted": int(valid.sum()),
        "Hybrid_fitted_unflagged": int(hy.status.eq("model_fit").sum()),
        "Hybrid_fitted_review": int(hy.status.eq("model_fit_review").sum()),
        "Hybrid_unresolved": int((~valid).sum()),
        "Hybrid_HLCE_near_boundary": int(hy.hlce_near_boundary.eq(True).sum()),
        "Hybrid_HLCE_weak_profile_identification": int(
            flags.str.contains("weakly identified").sum()
        ),
        "field_HCB_retained": True,
        "field_HCB_zero_records": int(hy.hcb.eq(0).sum()),
        "derived_volume_is_model_envelope_not_foliage_volume": True,
        "sensitivity_is_not_statistical_confidence_interval": True,
    }
    write_json(ROOT / "results/qa/structural_summary.json", receipt)
    print(receipt, flush=True)
