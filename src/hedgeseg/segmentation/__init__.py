"""Assign leaf-off ALS and leaf-on UAV points to surveyed hedgerow trees.

Module map (read in this order):

    scoring.py          pick the winning tree for each point; horizontal distance
    seeds.py            basal seeds near each surveyed stem (strict and wider window)
    stem_paths.py       shortest-path ("graph") scores through the ALS points
    canopy_peaks.py     canopy-height model, peaks, upper seeds for tall trees
    height.py           field-height penalty, 2 m height classes, selective requests
    transfer.py         pass ALS identities to UAV points
    combine.py          merge two assignments point by point (the combined method)
    baselines.py        nearest-stem proximity and canopy-height watershed
    methods.py          the eleven methods as recipes built from the pieces above
    pipeline.py         runs everything and writes the label files

See docs/METHODS.md for the physical meaning of every rule and constant.
"""
