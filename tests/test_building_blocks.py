"""Small, data-free tests of each building block (run in seconds)."""

import numpy as np
import pytest
from hedgeseg.segmentation.combine import combine_by_margin
from hedgeseg.segmentation.height import height_classes, height_penalty
from hedgeseg.segmentation.scoring import horizontal_distance, pick_lowest_cost
from hedgeseg.segmentation.seeds import add_wide_window_seeds, basal_seeds
from hedgeseg.segmentation.stem_paths import link_cost, stem_path_scores


def test_link_cost_prefers_short_steps():
    # the worked example of docs/METHODS.md
    assert link_cost(0.9) == pytest.approx(2.925)
    assert 3 * link_cost(0.3) == pytest.approx(1.125)


def test_pick_lowest_cost_and_margin():
    costs = np.array([[1.0, 5.0, np.inf], [3.0, 2.0, np.inf]])
    r = pick_lowest_cost(costs, np.array([10, 20]))
    assert r["label"].tolist() == [10, 20, 0]  # no feasible tree -> 0
    assert r["margin"].tolist() == [2.0, 3.0, 0.0]


def test_single_feasible_tree_has_zero_margin():
    costs = np.array([[1.0], [np.inf]])
    r = pick_lowest_cost(costs, np.array([1, 2]))
    assert r["label"][0] == 1 and r["margin"][0] == 0


def test_horizontal_distance_ignores_height():
    d = horizontal_distance(np.array([[3.0, 4.0, 100.0]]), np.array([[0.0, 0.0]]))
    assert d[0, 0] == pytest.approx(5.0)


def test_height_penalty_is_one_sided_and_quadratic():
    xyz = np.array([[0, 0, 3.0], [0, 0, 5.5], [0, 0, 6.5]])
    p = height_penalty(xyz, np.array([0.0]), np.array([4.0]), 0.5)[0]
    assert p[0] == 0 and p[1] == pytest.approx(2.0) and p[2] == pytest.approx(8.0)
    unconstrained = height_penalty(xyz, np.array([0.0]), np.array([np.nan]), 0.5)
    assert not unconstrained.any()


def test_height_classes():
    assert height_classes(np.array([0.2, 3.7, 4.9, 13.0])).tolist() == [0.5, 4.0, 4.0, 14.0]


def test_combine_keeps_sole_parent_and_uses_margin():
    g = np.array([1, 0, 1, 1])
    t = np.array([0, 2, 2, 2])
    gm = np.array([0.0, 0.0, 0.5, 0.2])
    tm = np.array([0.0, 0.0, 0.3, 0.2])
    label, took_g = combine_by_margin(g, t, gm, tm)
    assert label.tolist() == [1, 2, 1, 1]  # tie -> stem-path parent
    assert took_g.tolist() == [True, False, True, True]


def _upright_stem(x, y, n=6, top=1.8):
    z = np.linspace(0.4, top, n)
    return np.c_[np.full(n, x), np.full(n, y), z], z


def test_seed_needs_upright_group_near_stem():
    a, ha = _upright_stem(0.1, 0.0)
    flat = np.c_[np.full(5, 5.1), np.zeros(5), np.full(5, 1.0)]  # flat cluster at the 2nd stem
    xyz, h = np.r_[a, flat], np.r_[ha, np.full(5, 1.0)]
    seeds = basal_seeds(xyz, h, np.array([[0.0, 0.0], [5.0, 0.0]]))
    assert len(seeds[0]) == 6 and len(seeds[1]) == 0


def test_wide_window_recovers_high_stem_only_when_needed():
    xyz, h = _upright_stem(0.0, 0.0, top=4.0)
    xyz[:, 2] = h = np.linspace(2.2, 4.0, 6)  # nothing below 2 m
    strict = basal_seeds(xyz, h, np.array([[0.0, 0.0]]))
    assert len(strict[0]) == 0
    wide, used = add_wide_window_seeds(strict, xyz, h, np.array([[0.0, 0.0]]))
    assert used[0] and len(wide[0]) == 6


def test_stem_paths_follow_a_chain_of_points():
    # tree A: an upright stem then a branch reaching towards B; tree B: a stem
    stem_a, ha = _upright_stem(0.0, 0.0)
    branch = np.c_[np.linspace(0.3, 2.4, 8), np.zeros(8), np.full(8, 2.0)]
    stem_b, hb = _upright_stem(3.5, 0.0)
    xyz = np.r_[stem_a, branch, stem_b]
    h = np.r_[ha, branch[:, 2], hb]
    stems = np.array([[0.0, 0.0], [3.5, 0.0]])
    scores, seeds, info = stem_path_scores(xyz, h, stems, np.zeros(2))
    labels = pick_lowest_cost(scores, np.array([1, 2]))["label"]
    tip = 6 + 7  # last branch point: 2.4 m from A, 1.1 m from B
    nearest = pick_lowest_cost(horizontal_distance(xyz, stems), np.array([1, 2]))["label"]
    assert nearest[tip] == 2  # proximity would give it to B...
    assert labels[tip] == 1  # ...the path through the branch gives it to A
    assert info["reachable_fraction"] == 1.0
