# Hedgerow tree segmentation

Python tools for assigning ALS and UAV point clouds to surveyed trees in a mixed
hedgerow. The work covers segmentation, photographic evaluation and crown geometry.
Research code from INRAE (UMR PIAF, Clermont-Ferrand). The field data are private;
the public examples use invented coordinates and trees.

## Read the code

The main algorithm is in `src/hedgeseg/segmentation/`:

- `scoring.py`: distances and selection of the lowest-cost owner.
- `seeds.py`: small groups of ALS returns near each surveyed stem.
- `stem_paths.py`: sparse graph construction and shortest paths from the stems.
- `height.py`: field-height penalties and coarse height classes.
- `transfer.py`: transfer of ALS labels to the denser UAV cloud.
- `combine.py`: explicit policy for combining two assignments.
- `pipeline.py`: runs the method comparison and writes local outputs.

Start with [the synthetic example](examples/compare_assignments.py) and
[the methods note](docs/METHODS.md). These show the cost logic without requiring
access to the study site. `tests/test_building_blocks.py` checks its basic behaviour.

## Run the public example

Python 3.10 or later:

```bash
python -m pip install -r requirements-core.txt
python examples/compare_assignments.py
python -m pytest tests -q
python tools/check_public_files.py
```

Only NumPy, SciPy and pytest are needed for these checks. The example prints
nearest-stem and graph-based assignments for a small invented branch geometry.

## Full local workflow

```bash
python -m pip install -e .
python run.py --help
```

The full pipeline also needs GIS/image dependencies and the private inputs described
in [docs/DATA.md](docs/DATA.md). Copy `config/project.example.json` to
`config/project.json` locally, then restore the actual site configuration from your
private working archive. Nulls in the example are deliberate; they are not study coordinates.
Do not run the full pipeline with invented calibration parameters.

For an external working directory, set `HEDGESEG_ROOT` to a private folder containing
`data/source/` and `config/project.json`. `HEDGESEG_CONFIG` can select the configuration
file separately. The runner and stage outputs use this working root.

```bash
python run.py prepare
python run.py segment
python run.py evaluate
```

Structure extraction, camera diagnostics and viewer generation are available as separate
stages. Generated viewers embed point arrays, photographs or metadata: keep them private.
The source templates under `src/` contain placeholders, not filled research payloads.

## Limits

The methods use surveyed stems and, for some variants, field heights. They do not
solve individual-tree detection without an inventory. Photographic outlines constrain
2D projection agreement, not a complete 3D crown boundary. Connectivity through ALS
returns is evidence for assignment, not verified branch reconstruction. Method constants
need sensitivity checks before transfer to a different site.

[Development notes](docs/DEVELOPMENT.md) · [data privacy](docs/PRIVACY.md)

Code under the MIT licence; the bundled three.js keeps its own licence. No data licence is granted.
