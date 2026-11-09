"""
THE WHOLE HEDGEROW, CORRECTED, SMALL ENOUGH TO SEND.   python preparation/04_export_shared_clouds.py

TERMINAL ONLY. It opens the full 175 million point cloud.

Same corrections as EXPORT_section.py, applied over all 391 m instead of a
100 m stretch, and then thinned so the result can actually be uploaded.

--------------------------------------------------------------------------
WHY THINNING IS NEEDED, AND WHY IT COSTS LITTLE

Over the whole hedgerow the drone contributes roughly 56 million points, about
650 MB. That is not a file anyone can send.

The drone samples the canopy surface at about 4 800 points per square metre.
Nothing in this project resolves anything at that scale: the laser manages 18,
the segmentation grid works at 0.25 m, and the crown parameters are quoted to
a tenth of a metre. Keeping one representative point per 0.20 m box therefore
throws away no information that is used anywhere, and reduces the file by a
factor of roughly sixty.

The point kept in each box is an ACTUAL POINT, the one with the highest
confidence, not an average of its neighbours. Averaging would invent positions
that were never measured and would smooth the canopy surface.

The laser is left whole. At 18 points per square metre it is already small.

WHAT IS APPLIED, AND WHAT IS NOT

  laser Z    untouched. It defines the reference.
  drone Z    the offset plane subtracted, read from ennezat_dz.json.

Both clouds carry hag, height above the terrain model. Ground returns are
kept, so alignment can still be checked against a surface. No cleaning,
classification or filtering beyond the thinning.
"""

from pathlib import Path
import json, sys, time
import numpy as np, pandas as pd
from scipy import ndimage

from settings import OUT, ALS, UAV, INVENTORY as INV, OUTLINE, EXPORT_DIR as DEST  # config/preparation.json
CACHE = OUT/"ennezat_B1.npz"
DZF   = OUT/"ennezat_dz.json"

MODE = "upload"
# "local"   full resolution, and the files are named exactly as the 100 m
#           section package named them. Point the existing pilot at this
#           folder and it runs on the whole hedgerow with nothing edited.
#           About 650 MB. That is fine on your own disk and it is what the
#           pilot would have thinned for itself anyway.
#
# "upload"  one point kept per VOXEL box, about 15 MB, small enough to send.
#           Use this only for sharing. Do NOT run analysis on it: the pilot
#           does its own thinning and would be thinning twice.

PAD_D  = 15.0      # m either side of the axis, so crowns are whole
PAD_S  = 5.0       # m beyond each end
from settings import CAMERA_STATIONS as STATIONS   # x, y columns; None to skip
STATION_R = 10.0   # m radius kept around each camera station.
# The stations sit about 60 m from the hedgerow axis, far outside PAD_D, so
# without this the ground beneath the camera is not in the export at all --
# and that ground is what lets the camera height be checked independently.
VOXEL  = 0.20      # m, upload mode only. 0.40 is a further factor of eight
VOXEL_ALS = 0.0    # the laser is already sparse; never thinned
CHUNK  = 5_000_000
ALSO_CSV = False   # a plain text copy, readable without laspy. Large.

if MODE not in ("local", "upload"):
    raise SystemExit('MODE must be "local" or "upload"')
if MODE == "local":
    VOXEL = 0.0
    NAMES = dict(als="ennezat_ALS_section.laz", uav="ennezat_UAV_section.laz",
                 inv_sec="ennezat_inventory_section.gpkg",
                 inv_full="ennezat_inventory_full.gpkg",
                 outline="ennezat_hedge_outline(1).gpkg")
else:
    NAMES = dict(als="ennezat_ALS_full.laz", uav="ennezat_UAV_full.laz",
                 inv_sec="ennezat_inventory_section.gpkg",
                 inv_full="ennezat_inventory_full.gpkg",
                 outline="ennezat_hedge_outline.gpkg")

L = []
def P(s=""): print(s, flush=True); L.append(str(s))
def H(t): P(); P("="*76); P(t); P("="*76)

try:
    import laspy
except ImportError:
    sys.exit("laspy is required:  pip install laspy[lazrs]")
DEST.mkdir(parents=True, exist_ok=True)
t0 = time.time()

H("1. THE FRAME AND THE CORRECTION")
C = np.load(CACHE, allow_pickle=True)
cx, cy, u0, u1 = C["axis"]; v0, v1 = -u1, u0
S0, S1 = C["srange"]
dtm = C["dtm"]; gs0, gd0, DRES = C["dtm_org"]
DZI = json.loads(DZF.read_text())
MOD = DZI.get("model", dict(type="constant", c=float(DZI["dz"])))
def offset(x, y):
    if MOD["type"] == "plane":
        return MOD["ax"]*(x-MOD["x0"])+MOD["ay"]*(y-MOD["y0"])+MOD["c"]
    return np.full(np.shape(x), MOD["c"], dtype="float64")
def to_sd(x, y):
    dx, dy = x-cx, y-cy
    return dx*u0+dy*u1, dx*v0+dy*v1
def terrain(s, d):
    s_ = np.atleast_1d(np.asarray(s, "float64"))
    d_ = np.atleast_1d(np.asarray(d, "float64"))
    v = ndimage.map_coordinates(
        dtm, [np.clip((d_-gd0)/DRES, 0, dtm.shape[0]-1),
              np.clip((s_-gs0)/DRES, 0, dtm.shape[1]-1)],
        order=1, mode="nearest").astype("float64")
    return v if np.ndim(s) else float(v[0])
A0, A1 = S0-PAD_S, S1+PAD_S
P(f"hedgerow s = {S0:.1f} to {S1:.1f} m; exporting {A0:.1f} to {A1:.1f} m")
P(f"kept across the hedgerow: |d| <= {PAD_D:.1f} m")
P(f"offset model {MOD['type']}, representative {float(DZI['dz']):+.3f} m")
P(f"   subtracted from the drone; the laser is left as delivered")
cs = np.array([A0, A1, A1, A0]); cd = np.array([-PAD_D, -PAD_D, PAD_D, PAD_D])
BX, BY = cx+cs*u0+cd*v0, cy+cs*u1+cd*v1
STX = STY = None
if STATIONS and Path(STATIONS).exists():
    st = pd.read_csv(STATIONS)
    xc = next((c for c in st.columns if c.lower() in ("x", "x_l93", "easting")), None)
    yc = next((c for c in st.columns if c.lower() in ("y", "y_l93", "northing")), None)
    if xc and yc:
        STX = st[xc].to_numpy(float); STY = st[yc].to_numpy(float)
        ds = (STX-cx)*v0+(STY-cy)*v1
        P(f"{len(STX)} camera stations, {np.abs(ds).min():.0f} to "
          f"{np.abs(ds).max():.0f} m from the hedgerow axis")
        P(f"   keeping a {STATION_R:.0f} m radius around each, because they lie")
        P(f"   well outside |d| <= {PAD_D:.0f} m and the ground beneath the")
        P(f"   camera is needed to check its height independently")
        BX = np.r_[BX, STX-STATION_R, STX+STATION_R]
        BY = np.r_[BY, STY-STATION_R, STY+STATION_R]
else:
    P(f"no camera stations file at {STATIONS}; station ground not included")
XMIN, XMAX = BX.min()-2, BX.max()+2
YMIN, YMAX = BY.min()-2, BY.max()+2
P(f"bounding box {XMAX-XMIN:.0f} x {YMAX-YMIN:.0f} m in Lambert-93")

def clip(src, dst, apply_offset, vox, label):
    H(f"2. {label}")
    if not src.exists():
        P(f"NOT FOUND: {src}"); return 0
    fin = laspy.open(str(src)); hd = fin.header
    P(f"{src.name}")
    P(f"   {hd.point_count:,} points, format {hd.point_format.id}, "
      f"version {hd.version}")
    nh = laspy.LasHeader(point_format=hd.point_format.id,
                         version=str(hd.version))
    nh.scales, nh.offsets = hd.scales, hd.offsets
    ncrs = 0
    for v in hd.vlrs:
        if getattr(v, "user_id", "") == "LASF_Projection":
            nh.vlrs.append(v); ncrs += 1
    for d in getattr(hd.point_format, "extra_dimensions", []):
        try:
            nh.add_extra_dim(laspy.ExtraBytesParams(
                name=d.name, type=d.dtype,
                description=getattr(d, "description", "")))
        except Exception:
            pass
    HAG = True
    try:
        nh.add_extra_dim(laspy.ExtraBytesParams(
            name="hag", type=np.float32,
            description="height above terrain, m"))
    except Exception as e:
        HAG = False; P(f"   hag not added: {e}")
    # LAS allows an extra-byte description of at most 32 characters, so these
    # are terse. The README carries the full meaning.
    EXTRA = True
    try:
        nh.add_extra_dim(laspy.ExtraBytesParams(
            name="source_index", type=np.uint64,
            description="row in original file"))
        nh.add_extra_dim(laspy.ExtraBytesParams(
            name="voxel_count", type=np.uint32,
            description="points represented"))
        nh.add_extra_dim(laspy.ExtraBytesParams(
            name="roi", type=np.uint8,
            description="1=hedge 2=station ground"))
    except Exception as e:
        EXTRA = False; P(f"   traceability fields not added: {e}")
    keep = [d for d in hd.point_format.dimension_names
            if d in nh.point_format.dimension_names]
    P(f"   projection records carried over: {ncrs}")
    thin_txt = f"one point per {vox:.2f} m box" if vox > 0 else "none"
    P(f"   thinning: {thin_txt}")
    seen = set(); n = kept = 0
    put = None
    csv_rows = []
    with laspy.open(str(dst), mode="w", header=nh) as fout:
        for meth in ("write_points", "append_points", "write"):
            if hasattr(fout, meth): put = getattr(fout, meth); break
        for pts in fin.chunk_iterator(CHUNK):
            n += len(pts)
            x, y = np.asarray(pts.x), np.asarray(pts.y)
            q = (x >= XMIN)&(x <= XMAX)&(y >= YMIN)&(y <= YMAX)
            if not q.any(): continue
            idx0 = n-len(pts)
            src_idx = (np.flatnonzero(q)+idx0).astype(np.uint64)
            sub = pts[q]
            sx, sy = np.asarray(sub.x), np.asarray(sub.y)
            s_, d_ = to_sd(sx, sy)
            box = (s_ >= A0)&(s_ <= A1)&(np.abs(d_) <= PAD_D)
            roi = np.where(box, 1, 0).astype(np.uint8)   # 1 = hedgerow corridor
            if STX is not None:
                near = np.zeros(len(sx), bool)
                for tx, ty in zip(STX, STY):
                    near |= ((sx-tx)**2+(sy-ty)**2) <= STATION_R**2
                roi[near&~box] = 2                      # 2 = camera-station ground
                box = box|near
            if not box.any(): continue
            sub = sub[box]; s_, d_ = s_[box], d_[box]
            src_idx = src_idx[box]; roi = roi[box]
            z = np.asarray(sub.z, dtype="float64")
            if apply_offset:
                z = z-offset(np.asarray(sub.x), np.asarray(sub.y))
            if vox > 0:
                # One ACTUAL point per box, and where the file records a
                # confidence, the best one -- which is what the pilot's own
                # thinning does. It matters: the median confidence here is 2
                # of 24, so an arbitrary pick would keep more noise than
                # signal. No position is averaged or invented.
                key = np.floor(np.c_[np.asarray(sub.x), np.asarray(sub.y), z]
                               /vox).astype(np.int64)
                try:
                    order = np.argsort(-np.asarray(sub.confidence),
                                       kind="stable")
                except Exception:
                    order = np.arange(len(key))
                _, first, counts = np.unique(key[order], axis=0,
                                             return_index=True,
                                             return_counts=True)
                take = np.zeros(len(key), bool); take[order[first]] = True
                if not take.any(): continue
                vcount = np.zeros(len(key), np.uint32)
                vcount[order[first]] = counts.astype(np.uint32)
                vcount = vcount[take]
                sub = sub[take]; s_, d_, z = s_[take], d_[take], z[take]
                src_idx = src_idx[take]; roi = roi[take]
            else:
                vcount = np.ones(len(sub), np.uint32)
            new = laspy.LasData(nh, points=laspy.PackedPointRecord.zeros(
                len(sub), nh.point_format))
            for dn in keep:
                try: setattr(new, dn, getattr(sub, dn))
                except Exception: pass
            if apply_offset: new.z = z
            hg = (z-terrain(s_, d_)).astype("float32")
            if HAG: new.hag = hg
            if EXTRA:
                new.source_index = src_idx
                new.voxel_count = vcount
                new.roi = roi
            put(new.points)
            kept += len(sub)
            if ALSO_CSV:
                csv_rows.append(np.c_[np.asarray(new.x), np.asarray(new.y),
                                      np.asarray(new.z), hg,
                                      np.asarray(new.classification)])
            if n % (CHUNK*5) == 0:
                P(f"   ... {n:,} scanned, {kept:,} kept")
    mb = dst.stat().st_size/1e6
    P(f"   scanned {n:,}, kept {kept:,} "
      f"({kept/max(n,1)*100:.2f} %)")
    P(f"   written {dst.name}   {mb:.1f} MB")
    if ALSO_CSV and csv_rows:
        arr = np.vstack(csv_rows)
        f = dst.with_suffix(".csv")
        pd.DataFrame(arr, columns=["x", "y", "z", "hag", "classification"]
                     ).to_csv(f, index=False, float_format="%.3f")
        P(f"   written {f.name}   {f.stat().st_size/1e6:.1f} MB "
          f"(plain text, no laspy needed)")
    return kept

P(f"\nMODE = {MODE}")
if MODE == "local":
    P("   full resolution, section-style filenames. The existing pilot can")
    P("   be pointed at this folder unchanged and will run on all 391 m.")
else:
    P(f"   thinned to one point per {VOXEL:.2f} m box, for sending only.")
n_als = clip(ALS, DEST/NAMES["als"], False, VOXEL_ALS,
             "AIRBORNE LASER  (reference, Z untouched)")
n_uav = clip(UAV, DEST/NAMES["uav"], True, VOXEL,
             "DRONE  (offset subtracted)")

H("3. THE VECTOR FILES")
try:
    import geopandas as gpd
    def load_inventory(path, layer="stems"):
        g = gpd.read_file(path, layer=layer, fid_as_index=True).reset_index()
        for c in list(g.columns):
            if c.lower() in ("index", "fid") and "gpsID" not in g.columns:
                g = g.rename(columns={c: "gpsID"})
        if g.crs is not None and g.crs.to_epsg() != 2154:
            g = g.to_crs(epsg=2154)
        return g
    if INV.exists():
        G = load_inventory(INV)
        gs, gd = to_sd(G.geometry.x.values, G.geometry.y.values)
        G["hedge_s"], G["hedge_d"] = gs, gd
        for key in ("inv_sec", "inv_full"):
            G.to_file(DEST/NAMES[key], driver="GPKG", layer="stems")
        P(f"{NAMES['inv_sec']}   {len(G)} stems, CORRECTED IDs "
          f"{G.gpsID.min()}-{G.gpsID.max()}")
        P(f"{NAMES['inv_full']}   the same 194; over the whole hedgerow the")
        P(f"   'section' and 'full' inventories are identical, and both are")
        P(f"   written so the pilot finds the names it expects")
    else:
        P(f"{INV.name} not found; inventory not written")
    if OUTLINE.exists():
        for lyr in ("corridor", "centreline"):
            try:
                gpd.read_file(OUTLINE, layer=lyr).to_file(
                    DEST/NAMES["outline"], driver="GPKG", layer=lyr)
            except Exception:
                pass
        P(f"{NAMES['outline']}   corridor and centreline")
except ImportError:
    P("geopandas not available; vector files not written")

H("4. THE NOTE")
if VOXEL <= 0:
    RES_TXT = "Full resolution; no thinning was applied."
else:
    RES_TXT = (f"The drone is reduced to one ACTUAL point per {VOXEL:.2f} m "
               f"box, the highest-confidence point in each. It is a real "
               f"measurement, not an average, so no position is invented.")
(DEST/"README.md").write_text(f"""# Ennezat, the whole hedgerow, corrected

Same processing as the 100 m section, applied over all {A1-A0:.0f} m.

## Coordinate reference

Horizontal  RGF93 / Lambert-93, EPSG:2154
Vertical    NGF-IGN69, from ellipsoidal heights via RAF20

## Dates

Airborne laser   11 February 2023, LEAF-OFF
Drone            26 May 2026, LEAF-ON
Inventory        23 July 2026

Three years separate the laser from the drone. The offset below was measured
on the road surface, which does not grow, for that reason.

## What has been applied

| | vertical |
|---|---|
| laser | as delivered; it defines the reference |
| drone | offset plane subtracted, {float(DZI['dz']):+.3f} m at the hedgerow centre |

Both clouds carry `hag`, height above the terrain model built from laser
ground returns at 1 m.

## Resolution

{RES_TXT}

Drone: {n_uav:,} points. Laser: {n_als:,} points, whole.

Nothing is cleaned, reclassified or filtered. Ground returns are present.

## Extent

Hedgerow axis s = {A0:.1f} to {A1:.1f} m, |d| <= {PAD_D:.1f} m.
`hedge_s` runs along the hedgerow and `hedge_d` across it; the hedgerow bears
115.7 degrees.

## Files

| file | contents |
|---|---|
| `{NAMES['als']}` | laser, {n_als:,} points |
| `{NAMES['uav']}` | drone, {n_uav:,} points |
| `{NAMES['inv_sec']}` | 194 stems, corrected identifiers |
| `{NAMES['inv_full']}` | the same 194 |
| `{NAMES['outline']}` | delineated extent and centreline |

Filenames follow the 100 m section package, so a pipeline written for that
package runs on this one unchanged. Over the whole hedgerow the "section" and
"full" inventories are the same 194 stems.

## Per-point fields added

| field | meaning |
|---|---|
| `hag` | height above the terrain model, metres |
| `source_index` | row number of this point in the original file |
| `voxel_count` | how many original points this one represents; 1 if unthinned |
| `roi` | 1 = hedgerow corridor, 2 = ground within {STATION_R:.0f} m of a camera station |

The LAS format allows only 32 characters of description per field, so the
descriptions stored in the file itself are abbreviated. This table is the
full meaning.

Camera stations lie {'about 60' if STX is not None else '?'} m from the hedgerow
axis, well outside the |d| <= {PAD_D:.0f} m corridor, so the ground beneath each
camera is included separately and flagged `roi = 2`. It is there so a camera
height can be checked against a surface rather than assumed.
""", encoding="utf-8")
P("README.md written")

H("WHERE THEY ARE")
P(str(DEST))
tot = 0
for f in sorted(DEST.iterdir()):
    mb = f.stat().st_size/1e6; tot += mb
    P(f"   {f.name:<34}{mb:>9.1f} MB")
P(f"\ntotal {tot:.0f} MB      elapsed {time.time()-t0:.0f} s")
P()
P("If the drone file is still too large to send, raise VOXEL. At 0.40 m it")
P("falls by roughly a further factor of eight and the canopy surface is still")
P("finer than anything this project measures.")
(DEST/"REPORT_EXPORT_FULL.md").write_text(
    "# Full export\n\n```\n"+"\n".join(L)+"\n```\n", encoding="utf-8")
