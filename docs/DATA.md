# Private inputs

No research records are distributed in this repository. Restore the following inputs
from the private working archive if you are authorised to run the full pipeline:

| Local path under `data/source/` | Role |
|---|---|
| `clouds/ennezat_ALS_full.laz` | Leaf-off ALS returns |
| `clouds/ennezat_UAV_full.laz` | UAV surface cloud |
| `clouds/ennezat_hedge_outline.gpkg` | Study outline |
| `inventory/ennezat_inventory_full_clean.gpkg` | Corrected stem inventory and field measurements |
| `cameras/cameras_<season>_full.json` | Site-specific camera models |
| `cameras/camera_stations.csv` | Camera station measurements |
| `annotations/review.json` | Reviewed photographic outlines |
| `photos_display/` and photo manifests | Local photographs and display references |
| `review_viewer_support.json` | Payload used by the private review viewer |

See `src/hedgeseg/prepare.py`, `io.py`, and the stage loaders for the exact schema.
A local run can require additional supporting calibration files from the original archive.
The example configuration deliberately omits real coordinates and counts.
Do not infer missing calibration values from the synthetic example.
