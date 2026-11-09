"""Every numerical setting of the assignment methods, in one place.

Each value is listed with the physical quantity it stands for and its status:

* *inherited*  - set in the pilot study and not optimised;
* *tested*     - varied in a sensitivity analysis (see docs/METHODS.md);
* *empirical*  - a weight with no physical meaning of its own.

Lengths are metres. Changing a value here changes the results; the regression
test (tests/test_regression.py) will tell you by how much.
"""

# --- Basal seeds: where a stem should be visible below its crown ------------
SEED_MIN_HEIGHT = 0.35  # m above ground. Top of the ground/weed layer. inherited
SEED_MAX_HEIGHT = 2.0  # m. Trunk zone below most crown bases. inherited
SEED_MAX_HEIGHT_WIDE = 4.5  # m. Wider window, used only for trees with no strict seed. inherited
SEED_RADIUS = 0.85  # m from the surveyed stem. Search tolerance (~45 % of the
# 1.89 m median stem spacing), NOT a trunk radius. inherited
SEED_MIN_POINTS = 3  # a seed needs at least this many ALS returns. inherited
SEED_MAX_POINTS = 12  # keep the nearest/lowest 12. inherited
SEED_MIN_SPAN = 0.25  # m of vertical extent: upright stem, not a flat branch. inherited
SEED_HEIGHT_WEIGHT = 0.05  # ranking = horizontal distance + 0.05 x height. empirical

# --- Shortest paths through the ALS points ---------------------------------
LINK_MAX_LENGTH = 0.9  # m. Points closer than this are linked. tested: smallest
# value at which ~98.5 % of ALS points are reachable.
LINK_COST_SCALE = 0.6  # m. Cost of a link of length l is l * (1 + (l/0.6)^2),
# so long jumps across empty space are expensive. tested:
# weaker penalties scored higher; kept for reproducibility.
LINK_MIN_LENGTH = 1e-5  # m. Numerical guard for coincident points.
WEIGHT_DISTANCE = 0.35  # score = 0.35 x horizontal distance + 0.65 x path cost. empirical
WEIGHT_PATH = 0.65

# --- Canopy peaks (upper seeds for tall trees) -------------------------------
CHM_CELL = 0.25  # m. Canopy-height-model cell, about the UAV point spacing. inherited
CHM_FILL_MAX = 0.75  # m. Empty cells filled only this far from data. inherited
CHM_SMOOTH = 0.5  # m. Gaussian smoothing scale of the surface. inherited
PEAK_MIN_HEIGHT = 2.0  # m. Peaks lower than this are ignored. inherited
PEAK_MIN_SEPARATION = (
    2.0  # m. About one stem spacing; a peak is a surface feature, not a tree. inherited
)
PEAK_MATCH_DISTANCE = 3.0  # m. A peak is matched to an upper-stratum stem within 3 m. inherited
UPPER_SEED_RADIUS = 1.25  # m around the matched peak. inherited
UPPER_SEED_TOP_QUANTILE = 0.98  # local ALS "top" = 98th percentile height. inherited
UPPER_SEED_DEPTH = 1.0  # m. Seed points are taken from the upper 1 m below that top. inherited
UPPER_SEED_HEIGHT_WEIGHT = 0.15  # ranking = distance + 0.15 x depth below top. empirical
UPPER_SEED_MAX_POINTS = 8
UPPER_SEED_MAX_TOP_DIFFERENCE = 4.0  # m. ALS top and UAV peak must roughly agree. inherited
WATERSHED_MATCH_DISTANCE = 3.0  # m. Watershed comparator: basin peak to stem.

# --- Field-height constraint -------------------------------------------------
HEIGHT_PENALTY_COEFFICIENT = 2.0  # cost = 2 x (excess height)^2. empirical
HEIGHT_CLASS = 2.0  # m. Coarse variant rounds lower-tree heights to 2 m classes.
HEIGHT_CLASS_MINIMUM = 0.5  # m. Smallest rounded height.
REQUEST_WINDOW = 1.5  # m along the hedge: local canopy envelope for the selective variant.
REQUEST_MARGIN = 1.0  # m. Height is requested if a tree's top reaches within 1 m of it.
TOP_QUANTILE = 0.995  # supported top = 99.5th percentile height.
MIN_POINTS_FOR_TOP = 20

# --- Fallback screening -------------------------------------------------------
FALLBACK_TOP_QUANTILE = 0.99
FALLBACK_MAX_EXCESS = 1.5  # m. A recovered tree taller than field height + 1.5 m is rejected.
FALLBACK_MIN_POINTS = 20

# --- Transfer from ALS to UAV points ----------------------------------------
TRANSFER_DISTANCE_WEIGHT = (
    0.25  # UAV score = distance to tree's ALS + 0.25 x stem distance. empirical
)
TRANSFER_MIN_ALS_POINTS = 5  # a tree needs >= 5 ALS points to receive UAV points.
TRANSFER_MAX_SUPPORT = 4.0  # m from the tree's own ALS points. Active: excludes
# about half of candidate pairs within 8 m of a stem.
TRANSFER_MAX_STEM = 8.0  # m from the stem.
TRANSFER_BLOCK = 60_000  # points processed per block (memory only, no effect on results)

# --- Ground under each stem ------------------------------------------------
GROUND_NEIGHBOURS = 8  # stem ground = median of the 8 nearest ALS ground returns.
