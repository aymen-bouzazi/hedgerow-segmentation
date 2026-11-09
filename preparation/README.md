# preparation/

Raw IGN laser tile + 175 M-point drone cloud → the two shared files in `data/source/clouds/`.
Run in order, from the repository root, after creating `config/preparation.json`:

```bash
python preparation/01_read_and_terrain.py
python preparation/02_vertical_offset_road.py
python preparation/03_offset_constant_or_tilted.py
python preparation/04_export_shared_clouds.py
python preparation/05_check_heights_optional.py   # optional checks
```

Full explanation: [docs/PREPARATION.md](../docs/PREPARATION.md).
