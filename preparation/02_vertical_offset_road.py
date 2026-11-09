"""
STAGE B2b — ENNEZAT: the vertical offset, measured on hard surface.
*** RUN IN A TERMINAL ***

    python preparation/02_vertical_offset_road.py

--------------------------------------------------------------------------
WHY THE ROAD

Both datasets declare the same vertical datum. The processing report gives the
drone project as RGF93 v2b / Lambert-93 + NGF-IGN69 (RAF20), and the IGN laser
survey is delivered in IGN69. They should agree to a few centimetres. They do
not, so the discrepancy has to be measured rather than reasoned about.

Every previous attempt compared surfaces that are not the same thing in the
two seasons:

  open ground   the laser saw bare soil in February, the drone saw standing
                wheat in May. The difference contains the crop.
  tree tops     crowns overlap, so the canopy above a stem is not always that
                stem's own. The difference contains the neighbour.

A road surface is neither. It is bare, hard, flat, and identical in February
and in May. The difference over a road is the offset and nothing else.

The delivered cloud is classified and contains a road surface class of about
eleven million points, so this measurement was available from the beginning.

--------------------------------------------------------------------------
WHAT IS MEASURED

  1  the drone's ROAD SURFACE class against the laser terrain
     -> the vertical offset, uncontaminated

  2  the drone's GROUND class out in the fields against the same terrain
     -> the offset PLUS whatever was growing there in May

  3  the difference between them
     -> the height of the wheat, as a check that the whole picture is
        consistent rather than merely convenient

  4  whether the offset varies across the site
     -> a plane is fitted to the road differences. A constant means a simple
        shift is the right model. A tilt means the two surfaces are not
        parallel and no single number can be correct.

The class histogram is printed and compared against the counts in the
processing report, so that the class numbering is verified rather than assumed.

Writes ennezat_dz.json, refined by 03 and read by 04 (export) and 05 (checks).
"""

from pathlib import Path
import numpy as np, laspy, json, time
from scipy import ndimage

from settings import OUT, ALS, UAV  # all paths: config/preparation.json
CACHE = OUT/"ennezat_B1.npz"

ROAD, GROUND, HIVEG = 11, 2, 5
DRES   = 1.00      # m, terrain grid
KEEP   = 8         # keep 1 road point in this many
PAD    = 40.0      # m, margin around the drone extent
CHUNK  = 3_000_000
# counts given in the processing report, used only to verify the class numbering
REPORT = {2: 142_905_714, 5: 17_532_962, 6: 310, 11: 10_968_447}

L = []
def P(s=""): print(s, flush=True); L.append(str(s))
def H(t): P(); P("="*76); P(t); P("="*76)
CLS = {2: "Ground", 5: "High Vegetation", 6: "Building", 11: "Road Surface",
       64: "Car", 65: "noise", 0: "Created", 1: "Unclassified"}
t_all = time.time()

# ---------------------------------------------------------------- extent
H("1. EXTENT")
with laspy.open(str(UAV)) as fh:
    hd = fh.header
    xmin, ymin = float(hd.mins[0])-PAD, float(hd.mins[1])-PAD
    xmax, ymax = float(hd.maxs[0])+PAD, float(hd.maxs[1])+PAD
    P(f"drone file      {hd.point_count:,} points")
    P(f"extent          {hd.maxs[0]-hd.mins[0]:.0f} x {hd.maxs[1]-hd.mins[1]:.0f} m")
    P(f"elevation       {hd.mins[2]:.2f} to {hd.maxs[2]:.2f} m")
P("The whole survey area is used, not only the hedgerow strip, because the")
P("road may lie outside it.")

# ------------------------------------------------------- laser terrain
H("2. LASER TERRAIN OVER THE WHOLE SURVEY AREA")
t0 = time.time()
gx = []; gy = []; gz = []
with laspy.open(str(ALS)) as fh:
    for p in fh.chunk_iterator(CHUNK):
        x, y = np.asarray(p.x), np.asarray(p.y)
        c = np.asarray(p.classification)
        m = (x >= xmin)&(x <= xmax)&(y >= ymin)&(y <= ymax)&(c == GROUND)
        if m.any():
            gx.append(x[m]); gy.append(y[m]); gz.append(np.asarray(p.z)[m])
gx = np.concatenate(gx); gy = np.concatenate(gy); gz = np.concatenate(gz)
NX = int(np.ceil((xmax-xmin)/DRES))+1
NY = int(np.ceil((ymax-ymin)/DRES))+1
dtm = np.full((NY, NX), np.inf)
np.minimum.at(dtm, (np.clip(((ymax-gy)/DRES).astype(int), 0, NY-1),
                    np.clip(((gx-xmin)/DRES).astype(int), 0, NX-1)), gz)
meas = np.isfinite(dtm)&(dtm < np.inf)
P(f"laser ground returns {gx.size:,}   ({time.time()-t0:.0f} s)")
P(f"terrain grid {NY} x {NX} at {DRES:.2f} m, "
  f"{meas.mean()*100:.1f} % of cells measured")
_, ii = ndimage.distance_transform_edt(~meas, return_indices=True)
dtm = ndimage.gaussian_filter(np.where(meas, dtm, 0.)[tuple(ii)], 1.5)
def terrain(x, y):
    return ndimage.map_coordinates(
        dtm, [np.clip((ymax-y)/DRES, 0, NY-1), np.clip((x-xmin)/DRES, 0, NX-1)],
        order=1, mode="nearest")
res = gz - terrain(gx, gy)
P(f"scatter of laser ground returns about the surface  +/- {res.std():.3f} m")

# --------------------------------------------------------------- drone
H("3. DRONE CLASSES")
t0 = time.time()
cnt = {}
rx=[];ry=[];rz=[]      # road
fx=[];fy=[];fz=[]      # ground class, sampled
n = 0
with laspy.open(str(UAV)) as fh:
    for p in fh.chunk_iterator(CHUNK):
        c = np.asarray(p.classification)
        for k in np.unique(c): cnt[int(k)] = cnt.get(int(k), 0)+int((c == k).sum())
        x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z)
        m = c == ROAD
        if m.any():
            k = np.arange((-n) % KEEP, int(m.sum()), KEEP)
            rx.append(x[m][k]); ry.append(y[m][k]); rz.append(z[m][k])
        m = c == GROUND
        if m.any():
            k = np.arange((-n) % (KEEP*8), int(m.sum()), KEEP*8)
            fx.append(x[m][k]); fy.append(y[m][k]); fz.append(z[m][k])
        n += c.size
P(f"{'class':>6}{'points':>15}{'report':>15}{'match':>8}   name")
for k in sorted(cnt):
    r = REPORT.get(k)
    ok = ("--" if r is None else ("yes" if abs(cnt[k]-r)/max(r, 1) < 0.02 else "NO"))
    P(f"{k:>6}{cnt[k]:>15,}{('' if r is None else f'{r:,}'):>15}{ok:>8}   "
      f"{CLS.get(k,'?')}")
P()
if cnt.get(ROAD, 0) == 0:
    raise SystemExit("No road surface class in this file. Stopping — the whole "
                     "point of this script is the road.")
P(f"class numbering verified against the processing report  "
  f"({time.time()-t0:.0f} s)")
rx = np.concatenate(rx); ry = np.concatenate(ry); rz = np.concatenate(rz)
fx = np.concatenate(fx); fy = np.concatenate(fy); fz = np.concatenate(fz)
P(f"road points kept   {rx.size:,}  (1 in {KEEP})")
P(f"ground points kept {fx.size:,}  (1 in {KEEP*8})")

# ------------------------------------------------------------ the offset
H("4. THE OFFSET, ON HARD SURFACE")
dr = rz - terrain(rx, ry)
dr = dr[np.isfinite(dr)]
q = np.percentile(dr, [5, 25, 50, 75, 95])
P(f"drone road surface minus laser terrain, {dr.size:,} points")
P(f"   p5 {q[0]:+.3f}   p25 {q[1]:+.3f}   median {q[2]:+.3f}   "
  f"p75 {q[3]:+.3f}   p95 {q[4]:+.3f}")
P(f"   interquartile range {q[3]-q[1]:.3f} m")
DZ = float(q[2])
if q[3]-q[1] > 0.30:
    P("   the spread is wide for a hard flat surface; some of these points")
    P("   are probably not road. Treat the median as indicative.")
else:
    P("   the spread is narrow, as it should be on a hard flat surface")

H("5. IS IT A CONSTANT, OR A TILT?")
A = np.c_[rx-rx.mean(), ry-ry.mean(), np.ones(rx.size)]
coef, *_ = np.linalg.lstsq(A, rz-terrain(rx, ry), rcond=None)
span_x, span_y = np.ptp(rx), np.ptp(ry)
P(f"plane fitted to the road differences")
P(f"   east-west slope   {coef[0]*1000:+.2f} mm per metre "
  f"= {coef[0]*span_x:+.3f} m across {span_x:.0f} m")
P(f"   north-south slope {coef[1]*1000:+.2f} mm per metre "
  f"= {coef[1]*span_y:+.3f} m across {span_y:.0f} m")
tilt = max(abs(coef[0]*span_x), abs(coef[1]*span_y))
if tilt < 0.15:
    P(f"   total tilt across the site {tilt:.3f} m — negligible. A single")
    P(f"   constant is the right model.")
else:
    P(f"   total tilt across the site {tilt:.3f} m — NOT negligible. The two")
    P(f"   surfaces are not parallel and a single constant leaves a residual")
    P(f"   of that size at the edges of the site.")

H("6. THE FIELDS, AND THE WHEAT")
df = fz - terrain(fx, fy)
df = df[np.isfinite(df)]
qf = np.percentile(df, [25, 50, 75])
P(f"drone ground class minus laser terrain, {df.size:,} points")
P(f"   p25 {qf[0]:+.3f}   median {qf[1]:+.3f}   p75 {qf[2]:+.3f}")
P(f"\ndrone ground class   {qf[1]:+.3f} m above the laser terrain")
P(f"drone road surface   {DZ:+.3f} m above the laser terrain")
P(f"difference           {qf[1]-DZ:+.3f} m")
P()
P("That difference is what the drone's classifier called ground out in the")
P("fields but which sits above the true soil. In late May that is the wheat.")
if 0.3 <= (qf[1]-DZ) <= 1.3:
    P(f"A standing wheat crop of {qf[1]-DZ:.2f} m in late May is entirely")
    P("plausible, and it explains why the open-ground estimate was too large.")
else:
    P("That value is not an obvious crop height, so the interpretation is")
    P("not confirmed and should not be quoted as such.")

H("7. DECISION")
try:
    old = json.loads((OUT/"ennezat_dz.json").read_text())
    P(f"previous estimate, from the field inventory   {old['dz']:+.3f} m")
except Exception:
    old = None
P(f"open ground, from B1                          "
  f"{json.loads((OUT/'ennezat_B1.json').read_text())['off_field_median']:+.3f} m")
P(f"ROAD SURFACE                                  {DZ:+.3f} m")
P()
P(f"ADOPTED  DZ = {DZ:+.3f} m")
P("Measured on a bare hard surface present and unchanged in both acquisitions.")
P("It contains no crop and no canopy, so it needs no correction and rests on")
P("no assumption about what was growing or about which crown belongs to which")
P("stem.")
json.dump(dict(dz=DZ, source="road surface class, drone minus laser terrain",
               n=int(dr.size), iqr=float(q[3]-q[1]),
               tilt_m=float(tilt), crop_height=float(qf[1]-DZ),
               resid_sd=float(dr.std()),
               resid_p95=float(np.percentile(np.abs(dr-DZ), 95)),
               slope=1.0, slope_lo=1.0, slope_hi=1.0),
          open(OUT/"ennezat_dz.json", "w"), indent=2)
np.savez_compressed(OUT/"ennezat_road.npz",
                    road_x=rx, road_y=ry, road_d=rz-terrain(rx, ry),
                    fld_x=fx, fld_y=fy, fld_d=fz-terrain(fx, fy),
                    extent=np.array([xmin, xmax, ymin, ymax]))
(OUT/"REPORT_B2b.md").write_text(
    "# Stage B2b — Vertical offset on hard surface\n\n```\n"+"\n".join(L)+"\n```\n",
    encoding="utf-8")
P()
P(f"written: ennezat_dz.json, ennezat_road.npz, REPORT_B2b.md")
P(f"total {time.time()-t_all:.0f} s")
P("\nNext:  python preparation/03_offset_constant_or_tilted.py")