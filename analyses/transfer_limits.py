"""Do the 4 m / 8 m transfer limits change anything?   python analyses/transfer_limits.py

Re-runs the ALS-to-UAV transfer for the two wider-window parents with both
limits removed (ALS labels fixed) and counts how many winning UAV
assignments change. Writes results/qa/transfer_limits.json.
"""

import numpy as np
from _common import io
from hedgeseg.segmentation import constants as C
from hedgeseg.segmentation.pipeline import Hedge
from hedgeseg.segmentation.transfer import transfer_to_uav


def main():
    hedge = Hedge()
    stored = dict(np.load(io.WORKING / "hybrid_labels.npz"))
    out = {}
    for parent, heights in (("G", "lower"), ("T", "classes")):
        cap = hedge.caps(heights)
        args = (
            hedge.als_xyz,
            stored[f"{parent}_als_label"],
            hedge.uav_xyz,
            hedge.ids,
            hedge.stems_xy,
            hedge.ground_z,
            cap,
            0.5,
        )
        with_limits = transfer_to_uav(*args)["label"]
        saved = C.TRANSFER_MAX_SUPPORT, C.TRANSFER_MAX_STEM
        C.TRANSFER_MAX_SUPPORT = C.TRANSFER_MAX_STEM = 1e9
        try:
            without = transfer_to_uav(*args)["label"]
        finally:
            C.TRANSFER_MAX_SUPPORT, C.TRANSFER_MAX_STEM = saved
        out[parent] = {
            "changed_uav_points": int((with_limits != without).sum()),
            "share": float((with_limits != without).mean()),
        }
        print(parent, out[parent], flush=True)
    io.write_json(io.RESULTS / "qa" / "transfer_limits.json", out)


if __name__ == "__main__":
    main()
