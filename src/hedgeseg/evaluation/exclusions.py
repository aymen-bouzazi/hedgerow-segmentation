"""One place where the exclusion policy lives, and one table that states it.

THE POLICY
----------
A tree can be absent from a result for three different reasons, and conflating
them is how a surveyed tree quietly becomes a tree that does not exist.

  surveyed            it is in the field inventory. This never changes, and no
                      downstream failure removes a row from the survey.
  scorable            it has a hand-drawn crown outline, so a segmentation can
                      be scored against it. A tree with no outline is excluded
                      from SCORING only.
  parameterised       enough assigned points survived to fit crown geometry.

Failing to segment a tree is not evidence that the tree is absent. Tree 165 at
Ennezat was surveyed, never delineated, and holds no points under any method:
it stays in the inventory and in the structural tables, is excluded from every
accuracy figure, and is flagged here with the reason.

Reporting coverage and accuracy in the same number hides exactly this. The
receipt keeps them apart: how many trees could be scored at all, and how well
the scorable ones did.

PORTING
-------
The policy is data-driven, not hard-coded to any gpsID. A new site inherits it
by having an outline or not. If a site needs a tree dropped for some other
reason, add it to `manual_exclusions` in config/project.json with a written
reason; it will appear here with that reason rather than vanishing.
"""

import numpy as np
import pandas as pd

from ..io import ROOT, config, labels, write_json
from .context import reference_inputs, result_root


def run(review_path=None, output_dir=None):
    """Stage ``exclusions``: one row per tree saying whether it is surveyed, scorable and parameterised, and why not."""
    cfg = config()
    inv = pd.read_csv(ROOT / "data/working/inventory.csv")
    ann, cams, selected, review_receipt = reference_inputs(review_path)
    outlined = {int(a["gpsID"]) for a in ann["annotations"] if a.get("type") == "visible_crown"}
    accepted = {a["gpsID"] for a in selected}
    dest = result_root(output_dir)
    L = labels()
    manual = {int(k): v for k, v in (cfg.get("manual_exclusions") or {}).items()}

    params_path = ROOT / "results/parameters/parameters_all_methods.csv"
    params = pd.read_csv(params_path) if params_path.exists() else None

    rows = []
    for r in inv.itertuples():
        g = int(r.gpsID)
        held = {m: int((lab == g).sum()) for m, lab in L.items()}
        prod = held.get("Hybrid", 0)
        reasons = []
        if g in manual:
            reasons.append(f"manually excluded: {manual[g]}")
        if g not in outlined:
            reasons.append("no hand-drawn crown outline, so no reference to score against")
        if prod == 0:
            reasons.append("holds no points under the production method")
        par = np.nan
        if params is not None:
            q = params[(params.method == "Hybrid") & (params.gpsID == g)]
            par = bool(len(q) and str(q.iloc[0].status).startswith("model_fit"))
        rows.append(
            dict(
                gpsID=g,
                stratum=getattr(r, "stratum", None),
                species_field=getattr(r, "variety", None),
                surveyed=True,
                has_outline=g in outlined,
                scorable=g in accepted,
                production_points=prod,
                methods_holding_points=int(sum(1 for v in held.values() if v > 0)),
                parameterised=par,
                exclusion_reason="; ".join(reasons),
            )
        )

    x = pd.DataFrame(rows)
    x.to_csv(dest / "tables/tree_status.csv", index=False)

    excluded = x[~x.scorable]
    receipt = dict(
        surveyed=int(len(x)),
        scorable=int(x.scorable.sum()),
        excluded_from_scoring=int((~x.scorable).sum()),
        excluded_ids=excluded.gpsID.tolist(),
        excluded_reasons={int(r.gpsID): r.exclusion_reason for r in excluded.itertuples()},
        holding_no_points_under_production=int((x.production_points == 0).sum()),
        parameterised=int(x.parameterised.sum()) if x.parameterised.notna().any() else None,
        policy=(
            "Surveyed trees are never removed from the inventory. A tree "
            "without a crown outline is excluded from accuracy figures "
            "only, because there is nothing to score it against. Failing "
            "to segment a tree is not evidence that the tree is absent."
        ),
        coverage_vs_accuracy=(
            "Coverage (how many trees could be scored at all) "
            "and accuracy (how well the scorable ones did) are "
            "reported separately and must not be combined into "
            "one figure."
        ),
    )
    receipt["review_inputs"] = review_receipt
    receipt["parameterised_definition"] = (
        "all required directional geometry fitted; Htot alone is insufficient"
    )
    write_json(dest / "qa/exclusions.json", receipt)

    print(
        f"Tree status: {len(x)} surveyed, {int(x.scorable.sum())} scorable, "
        f"{int((~x.scorable).sum())} excluded from scoring",
        flush=True,
    )
    for r in excluded.itertuples():
        print(f"  gpsID {r.gpsID}: {r.exclusion_reason}", flush=True)
