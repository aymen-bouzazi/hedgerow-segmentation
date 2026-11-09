"""An invented branch that reaches towards its neighbour's stem.

No site measurements or research arrays are loaded. Run from any directory.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hedgeseg.segmentation.scoring import horizontal_distance, pick_lowest_cost
from hedgeseg.segmentation.stem_paths import stem_path_scores


def main():
    z = np.linspace(0.4, 1.8, 6)
    stem_a = np.column_stack([np.zeros(6), np.zeros(6), z])
    stem_b = stem_a + [3.5, 0, 0]
    branch = np.column_stack([np.linspace(0.3, 2.4, 8), np.zeros(8), np.full(8, 2.0)])
    xyz = np.concatenate([stem_a, branch, stem_b])
    stems = np.array([[0.0, 0.0], [3.5, 0.0]])
    ids = np.array([1, 2])
    nearest = pick_lowest_cost(horizontal_distance(xyz, stems), ids)["label"]
    costs, _, _ = stem_path_scores(xyz, xyz[:, 2], stems, np.zeros(2))
    graph = pick_lowest_cost(costs, ids)["label"]
    tip = len(stem_a) + len(branch) - 1
    print("Synthetic branch-tip owner:")
    print("  nearest stem:", nearest[tip])
    print("  stem paths:  ", graph[tip])
    print("The path follows the branch back to tree 1 in this constructed example.")


if __name__ == "__main__":
    main()
