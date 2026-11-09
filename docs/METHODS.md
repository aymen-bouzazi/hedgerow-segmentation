# Assignment methods

The input is a set of surveyed stem positions, height information where available,
and two point clouds: leaf-off ALS and a denser UAV canopy surface.
The output is a tree identifier for each assignable point. Zero means unassigned.

## Costs and ownership

Each method builds a matrix with one row per tree and one column per point.
A small value indicates a more plausible owner; infinity excludes that tree.
`pick_lowest_cost` chooses the best row. The margin is the second-best cost minus
the best cost, with zero when there is no finite competitor. It is not a probability.

Nearest-stem assignment uses horizontal distance alone. It provides a comparator,
but a branch may extend closer to another tree's stem than to its own.

## Stem paths

`basal_seeds` searches near each stem for an upright cluster of low ALS returns.
Candidate points are first allocated to the nearest stem, so seed groups do not overlap.
Seed radius, minimum point count and vertical span are in `segmentation/constants.py`.

`stem_path_scores` connects nearby ALS returns using a sparse graph. An edge of
length l has cost `l * (1 + (l / LINK_COST_SCALE)**2)`. Longer jumps therefore cost
more than several short steps across the same total distance. Virtual nodes connect
the surveyed stems to their seeds. Dijkstra's algorithm calculates route costs.
The final score combines horizontal stem distance and route cost using explicit weights.

This is a competition between costs, not a separation into disconnected trees.
Intertwined crowns can be connected and a return is not automatically verified wood.

## Height constraints and alternative seeds

A soft, one-sided quadratic penalty discourages assigning a high point to a lower tree.
Trees without known heights remain unconstrained. Coarse height classes test whether
rough field information is sufficient. Some variants request a field height when a
lower tree's assigned top approaches the local canopy envelope.

A wider seed window retries trees without usable low seeds. Other variants add seeds
near upper-canopy peaks. These changes address missing trunk returns but can also
capture a neighbour, so they require evaluation rather than automatic acceptance.

## UAV transfer and combination

ALS identities are transferred to UAV points using the support and distance rules in
`transfer.py`. `combine_by_margin` keeps the sole parent's label where only one parent
assigns a point. When both assign it, the larger margin wins; ties favour the stem-path
parent. Margins from different methods are not calibrated on a common probability scale.

## Evaluation and structure

The photographic stage projects point assignments through camera models and compares
them with reviewed visible-crown outlines. Camera alignment, occlusion and annotation
choices affect those scores. The 2D reference does not establish absolute 3D accuracy.
Structure modules estimate supported heights and crown envelopes, with flags for weak
support. An envelope volume is not a direct measurement of occupied foliage volume.

The public tests exercise small, invented geometries. They do not reproduce field scores
because clouds, inventory, camera calibration, annotations and reference tables are private.
