"""Run the Ennezat pipeline:  python run.py <stage>   (or  python run.py all)

Stages, in order (each one only needs the stages before it):

  prepare      shared clouds -> analysis point set        (~10 s)
  segment      eleven assignments of points to trees     (~1 min)
  evaluate     agreement with the photo outlines (F1)    (~40 s)
  parameters   height, crown base, radii, widest level   (~3 min)
  structure    crown volumes, 3D occupancy, hedge height, conflict and
               exclusion diagnostics, review queue        (~4 min)
  viewers      offline HTML viewers in ./viewers          (~2 min)
  check        consistency checks on all outputs

Optional: cameras (replay the July camera fit; check road residuals).
Every stage appends a line to results/qa/execution.jsonl.
"""

import argparse
import importlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hedgeseg import io  # noqa: E402 -- local src path is added above

STAGES = {
    "prepare": "hedgeseg.prepare",
    "segment": "hedgeseg.segmentation.pipeline",
    "evaluate": "hedgeseg.evaluation.photographs",
    "parameters": "hedgeseg.structure.parameters",
    "exclusions": "hedgeseg.evaluation.exclusions",
    "conflict": "hedgeseg.evaluation.conflict",
    "occupancy": "hedgeseg.evaluation.carve3d",
    "volumes": "hedgeseg.structure.volumes",
    "solids": "hedgeseg.structure.solids",
    "hedge-height": "hedgeseg.structure.hedge_height",
    "review-queue": "hedgeseg.structure.quality",
    "viewers": "hedgeseg.viewers.build",
    "camera-replay": "hedgeseg.cameras.calibration_replay",
    "road-residuals": "hedgeseg.cameras.registration",
    "check": "hedgeseg.checks",
}
GROUPS = {
    "structure": [
        "exclusions",
        "conflict",
        "occupancy",
        "volumes",
        "solids",
        "hedge-height",
        "review-queue",
    ],
    "cameras": ["camera-replay", "road-residuals"],
}
GROUPS["all"] = [
    "prepare",
    "segment",
    "evaluate",
    "parameters",
    *GROUPS["structure"],
    "viewers",
    "check",
]
GROUPS["core"] = ["prepare", "segment", "evaluate", "parameters"]
# stages that accept an alternative review file / output folder
REVIEWABLE = {"evaluate", "exclusions", "conflict", "occupancy", "solids"}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=[*STAGES, *GROUPS])
    parser.add_argument(
        "--review-json",
        type=Path,
        help="alternative reviewed outlines/cameras exported from the review viewer",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="write diagnostics to a separate results folder"
    )
    args = parser.parse_args()
    stages = GROUPS.get(args.stage, [args.stage])
    for folder in (
        "qa",
        "tables",
        "parameters",
        "segmentation",
        "structure",
        "diagnostics",
        "point_clouds",
        "calibration",
    ):
        (io.ROOT / "results" / folder).mkdir(parents=True, exist_ok=True)
    log = io.ROOT / "results" / "qa" / "execution.jsonl"
    for stage in stages:
        start = time.time()
        print(f"\n=== {stage}", flush=True)
        record = {"stage": stage, "python": platform.python_version()}
        try:
            module = importlib.import_module(STAGES[stage])
            if stage in REVIEWABLE:
                module.run(args.review_json, args.output_dir)
            else:
                module.run()
            record.update(status="passed")
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            record.update(
                seconds=round(time.time() - start, 1),
                completed_utc=datetime.now(timezone.utc).isoformat(),
            )
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
