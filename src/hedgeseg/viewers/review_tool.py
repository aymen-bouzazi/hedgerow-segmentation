"""Restore the established photo review interface, backed by canonical arrays.

The original UI is a source template. This builder updates its data, retains
annotation operations and adds reversible orientation fitting/XYZ inspection.
No point coordinates, inventory offsets or annotation vertices are modified.
"""

import base64
import json

import numpy as np
from scipy.spatial import cKDTree

from ..io import ROOT, annotations, cameras, clouds, inventory, labels, sha, write_json
from . import TEMPLATES


def pack(a, dtype):
    """Encode an array as base64 for embedding in HTML."""
    return base64.b64encode(np.asarray(a, dtype=dtype).tobytes()).decode("ascii")


def run():
    """Build review_full_hedgerow.html (photo review and outline editing tool)."""
    d, inv, labs, cams = clouds(), inventory(), labels(), cameras()
    support = json.loads((ROOT / "data/source/review_viewer_support.json").read_text())
    p = support
    p["version"] = "ennezat-chapter2-restored-review-v2"
    p["initial_confirmations"] = annotations()
    from ..cameras.calibration_replay import merge_ties

    merged, historical_sources = merge_ties(p["initial_confirmations"].get("tie_points", []))
    p["initial_confirmations"]["tie_points"] = merged
    p["initial_confirmations"]["historical_calibration_sources"] = historical_sources
    p["original_annotations_sha256"] = sha(ROOT / "data/source/annotations/review.json")
    p["preview_correction_m"] = 0
    p["fit_residuals"] = {}
    p["road_registration_diagnostic"] = {"additional_shift_applied_m": 0}
    old_methods = {m["name"]: m for m in p["methods"]}
    p["methods"] = []
    p["labels"] = {}
    six = dict(np.load(ROOT / "data/working/six_method_labels_full.npz"))
    hybrid = dict(np.load(ROOT / "data/working/hybrid_labels.npz"))
    for method, y in labs.items():
        meta = old_methods.get(
            method,
            {
                "name": method,
                "why": {
                    "Hybrid": "Choose between the two fallback graph assignments using their score separation.",
                    "G_fallback": "Basal ALS graph with a wider seed window for otherwise unsupported stems.",
                    "T_fallback": "Canopy-seeded ALS graph with fallback support for otherwise unsupported stems.",
                    "Nearest_stem_XY": "Planimetric nearest surveyed stem baseline.",
                    "Watershed": "Canopy-height watershed baseline; misses many lower crowns.",
                }.get(method, method),
                "inventory": "Stem positions. Graph fallback variants also use selective height information; see Methods.",
            },
        )
        if method == "T_targeted_1":
            meta = dict(
                meta,
                inventory="XY, stratum and 48 selectively requested lower-tree heights in the verified run.",
            )
        p["methods"].append(meta)
        als = six.get(method + "_als_label")
        if method == "Hybrid":
            als = hybrid["hybrid_als_label"]
        if method == "G_fallback":
            als = hybrid.get("G_als_label")
        if method == "T_fallback":
            als = hybrid.get("T_als_label")
        # Do not fabricate ALS assignments for methods which only export UAV labels.
        if als is None:
            als = np.zeros(len(d["als_xyz"]), dtype=np.uint16)
        p["labels"][method] = {
            s: {"label": pack(v, "<u2"), "candidate": pack(v, "<u2")}
            for s, v in [("uav", y), ("als", als)]
        }
        meta["als_labels_available"] = bool(np.any(als))
    p["clouds"] = {}

    def add(name, xyz, hag=None, source=None):
        p["clouds"][name] = {
            "xyz": pack(xyz, "<f8"),
            "count": len(xyz),
            "hag": pack(np.full(len(xyz), np.nan) if hag is None else hag, "<f4"),
            "source_index": pack(np.arange(len(xyz)) if source is None else source, "<f8"),
        }

    add("als", d["als_xyz"], d["als_hag"], d["als_source_index"])
    add("uav", d["uav_xyz"], d["uav_hag"], d["uav_source_index"])
    ground = np.c_[d["ground_xy"], d["ground_z"]]
    # Display subset only: evaluation and all source ground points are unchanged.
    add("ground", ground[::5], np.zeros(len(ground[::5])), np.arange(len(ground))[::5])
    p["ground_display_stride"] = 5
    # A display-only interpolation; not the upstream height-normalisation model.
    xx = np.arange(ground[:, 0].min(), ground[:, 0].max(), 1.0)
    yy = np.arange(ground[:, 1].min(), ground[:, 1].max(), 1.0)
    xy = np.array(np.meshgrid(xx, yy)).reshape(2, -1).T
    dist, ix = cKDTree(ground[:, :2]).query(xy, k=4)
    keep = dist[:, 0] <= 1.5
    w = 1 / np.maximum(dist[keep], 0.02) ** 2
    terrain = np.c_[xy[keep], (w * ground[ix[keep], 2]).sum(1) / w.sum(1)]
    add("terrain", terrain)
    road = np.empty((0, 3))
    try:
        import laspy

        src = next((ROOT / "data/source/clouds").glob("*UAV*full*.laz"))
        a = laspy.read(src)
        k = np.asarray(a.classification) == 11
        road = np.c_[a.x[k], a.y[k], a.z[k]]
    except StopIteration:
        pass
    add("road", road)
    p["stems_world"] = json.loads(
        inv[["gpsID", "x", "y", "ground_z", "Hsup"]].to_json(orient="records")
    )
    for f in p["frames"]:
        f["camera"] = dict(cams[f["id"]])
        f["image"] = "assets/photos/" + f["id"]
        f["quality"] = (
            "Supplied provisional camera. Check independent visible features before evaluating outlines."
        )
        f["metric"] = (
            "Native pixel coordinates; cloud XYZ is retained. A projected outline does not determine hidden 3D ownership."
        )
        for s in [
            "als",
            "uav",
            "ground",
            "terrain",
            "uav_low",
            "road",
            "uav_low_adjusted",
            "road_adjusted",
        ]:
            f[s] = {"count": 0, "index": "", "uv": ""}
        f["controls"] = []
    template = (TEMPLATES / "review_original.html").read_text()
    start = template.index("'use strict'")
    template = template[:start] + "__SCRIPT__</script></body></html>"
    template = template.replace("Compare six methods", "Compare segmentation methods")
    template = template.replace(
        "Photo correspondences · April / July", "Seasonal feature correspondences"
    )
    template = template.replace(
        "All 110 March base clicks and 44 April–July feature pairs are retained.",
        f"All 110 March base clicks and {len(merged)} confirmed seasonal feature pairs are retained.",
    )
    template = template.replace(
        "Your 44 confirmed pairs at stations 24–28 are preserved. Elsewhere suggestions are unconfirmed.",
        f"{len(merged)} saved feature pairs across 20 stations are preserved. Automatic suggestions require confirmation.",
    )
    template = template.replace(
        "Find the same fixed detail in April and July:",
        "Find the same fixed detail in the selected seasons:",
    )
    template = template.replace(
        "ALS terrain model · optional", "ALS ground interpolation · display only"
    )
    template = template.replace("Accepted · unassigned grey", "Assignments · unassigned grey")
    template = template.replace(
        '<option value="candidate">Best candidates · includes weak guesses</option>', ""
    )
    template = template.replace(
        '<option value="pan">Inspect / pan</option>',
        '<option value="pan">Inspect / pan</option><option value="inspect_xyz">Inspect point XYZ / depth</option>',
    )
    template = template.replace(
        '<select id="method"></select>',
        '<select id="method"><option value="none">None · unsegmented ALS / UAV</option></select>',
    )
    template = template.replace(
        '<p class="hint" id="methodWhy">',
        '<label><input id="outlineFilter" type="checkbox"> Show only the selected tree’s outlines</label><p class="hint" id="methodWhy">',
    )
    template = template.replace(
        "<p>No field HCB, HLCE, upper-tree Htot, species, DBH or trunk count controls these six point assignments.</p>",
        "<p>Field HCB and HLCE do not enter segmentation. Some variants use lower-tree heights; fallback support also consults field heights for some upper trees. These inputs are not independent validation targets.</p>",
    )
    template = template.replace(
        "No crown accuracy or winning method is claimed.",
        "The separate evaluation tables report 2D agreement, not independent 3D accuracy.",
    )
    template = template.replace(
        "compare the six frozen segmentations", "compare the segmentation methods"
    )
    template = template.replace(
        "</aside>", (TEMPLATES / "review_calibration_controls.html").read_text() + "</aside>"
    )
    template = template.replace(
        '<section id="tieControls" hidden>',
        '<section id="tieControls" hidden><label>Source season</label><select id="tieSource"><option value="april">April</option><option value="march">March</option><option value="july">July</option></select><label>Target season</label><select id="tieTarget"><option value="july">July</option><option value="april">April</option><option value="march">March</option></select>',
    )
    template = template.replace(
        "April on the left · July on the right.",
        "Selected source on the left · target on the right.",
    )
    template = template.replace(
        "Editing or adding a pair needs a later camera refit; it does not update live.",
        "Pairs are observations. Use Preview seasonal fit, inspect the result, then Apply; saving a pair alone does not move a camera.",
    )
    js = (TEMPLATES / "review_original.js").read_text()
    js = js.replace("$('method').value='T_none'", "$('method').value='none'")
    js = js.replace(
        "['als','uav'].includes(s)?DATA.labels[$('method').value][s][$('assignment').value]:null",
        "['als','uav'].includes(s)&&$('method').value!=='none'?DATA.labels[$('method').value][s].label:null",
    )
    js = js.replace("mode==='compare'?colour(id)", "mode==='compare'&&lab?colour(id)")
    js = js.replace(
        "drawAnno(a,false,u);",
        "if($('outlineFilter').checked&&+$('tree').value&&a.gpsID!==+$('tree').value)continue;drawAnno(a,false,u);",
    )
    js = js.replace(
        "$('methodControls').hidden=mode!=='compare'",
        "$('methodControls').hidden=mode==='reference'",
    )
    js = js.replace(
        "const m=DATA.methods.find(m=>m.name===$('method').value);$('methodWhy').textContent=m.why;$('inventory').textContent=m.inventory;",
        "const m=DATA.methods.find(m=>m.name===$('method').value);$('methodWhy').textContent=m?m.why:'Unsegmented selected cloud: no tree-assignment filter.';$('inventory').textContent=m?m.inventory:'No method selected.';",
    )
    js = js.replace(
        "if(['pan','edit','draft_box'].includes(t))return;forceReference();",
        "if(t==='inspect_xyz'){inspectPoint(p);return;}if(['pan','edit','draft_box'].includes(t))return;forceReference();",
    )
    js = js.replace(
        "if($('tool').value!=='pan')forceReference();",
        "if(!['pan','inspect_xyz'].includes($('tool').value))forceReference();",
    )
    js = js.replace(
        "(side==='left'?'april':'july')",
        "(side==='left'?$('tieSource').value:$('tieTarget').value)",
    )
    js = js.replace(
        "ties.filter(t=>t.station_id===frame.station_id)",
        "ties.filter(t=>t.station_id===frame.station_id&&t.source_photo_id===pairs.left?.frame.id&&t.target_photo_id===pairs.right?.frame.id)",
    )
    # Do not display residuals computed under an older camera as current diagnostics.
    js = js.replace("r=DATA.fit_residuals[t.uid]", "r=null")
    js = js.replace(
        "changeFrame();resize();persist();",
        (TEMPLATES / "review_extensions.js").read_text() + "\nchangeFrame();resize();persist();",
    )
    js = js.replace(
        "$('loadSuggestion').onclick=()=>{const s=",
        "$('loadSuggestion').onclick=()=>{if($('tieSource').value!=='april'||$('tieTarget').value!=='july'){say('Automatic suggestions are available for April to July only. Click a matching feature in each selected photograph.');return;}const s=",
    )
    text = template.replace(
        "__PAYLOAD__", json.dumps(p, separators=(",", ":"), allow_nan=False)
    ).replace("__SCRIPT__", js)
    (ROOT / "viewers/review_full_hedgerow.html").write_text(text, encoding="utf-8")
    write_json(
        ROOT / "results/qa/restored_review_build.json",
        {
            "frames": len(p["frames"]),
            "outlines": sum(
                a.get("type") == "visible_crown" for a in p["initial_confirmations"]["annotations"]
            ),
            "method_count": len(labs),
            "feature_pairs": len(merged),
            "historical_calibration_sources": historical_sources,
            "extra_uav_shift_m": 0,
            "annotation_source_sha256": p["original_annotations_sha256"],
            "base_fit_model": "orientation only; supplied camera centre and focal length fixed",
            "seasonal_fit_model": "rotation transfer, approximate common optical centre",
            "raw_mode": "unsegmented selected clouds, not all points in the source tile",
        },
    )
    print("Restored original review interface:", len(p["frames"]), "frames", flush=True)
