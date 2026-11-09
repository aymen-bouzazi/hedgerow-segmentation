"""
STAGE B1 — ENNEZAT: read the data and build the terrain.
*** RUN IN A TERMINAL ***

    python preparation/01_read_and_terrain.py

--------------------------------------------------------------------------
RULE THIS SCRIPT OBEYS, AND THE REASON FOR IT

Everything written by this script is an ELEVATION, in metres above the
vertical datum of the file it came from. Nothing is converted to a height
above ground, and no offset between the two datasets is applied.

The reason is a mistake made earlier. A figure plotted the laser surface as a
height above its own terrain, and the drone surface as a height above the
laser's terrain with no correction between them, on one pair of axes. The two
were in different vertical references and the plot appeared to show a five
metre disagreement that was mostly bookkeeping.

The structural fix is that the reading step never produces a height, so no
later step can accidentally mix a corrected quantity with an uncorrected one.
Heights are created once, in B2, after the offset has been decided, and every
height that exists after that point is in the same reference.

--------------------------------------------------------------------------
WHAT IT DOES

  1  Defines the hedgerow axis from the surveyed stems and the coordinate
     system (s, d): distance along the hedge and distance to one side of it.

  2  Audits the laser file: classes, returns, density.

  3  Builds the terrain from the laser ground class, and reports the scatter
     of ground returns about it. That scatter is the terrain model's own
     uncertainty and every canopy height computed later inherits it.

  4  Audits the drone file: classes, colour, presence of the Metashape
     confidence field.

  5  Builds the drone surfaces as elevations: highest and lowest elevation per
     cell, and the point count per cell.

  6  Measures the vertical offset between the two datasets by three
     independent routes and REPORTS them. It applies none of them. The choice
     is made in B2.

WRITES ennezat_B1.npz and REPORT_B1.md
"""

from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, laspy, json, time, sys
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from settings import OUT, STEMS, ALS, UAV  # all paths: config/preparation.json

BUF     = 25.0     # m, half-width of the strip read
HEDGE   = 4.0      # m, half-width called "the hedge" in summary statistics
FIELD   = (12.0, 20.0)   # m, the open ground used for the offset
DRES    = 1.00     # m, terrain grid
CRES    = 0.20     # m, surface grid
RSTEM   = 0.75     # m, radius searched for a stem's own canopy top
NEIGH   = 3.0      # m, a taller stem this close means the crown is overtopped
WINLEN  = 60.0
WINCAP  = 2_000_000
STRIDE  = 120
CHUNK   = 3_000_000

A, L = {}, []
def P(s=""): print(s, flush=True); L.append(str(s))
def H(t): P(); P("="*76); P(t); P("="*76)
CLS = {0:"never classified", 1:"unclassified", 2:"ground", 3:"low vegetation",
       4:"medium vegetation", 5:"high vegetation", 6:"building", 7:"low point",
       9:"water", 11:"road surface", 18:"high point", 64:"noise", 65:"noise",
       66:"reserved"}
t_all = time.time()

# ======================================================== 1. reference frame
H("1. REFERENCE FRAME")
g = gpd.read_file(STEMS)
if g.crs is not None and g.crs.to_epsg() != 2154: g = g.to_crs(2154)
PX, PY = g.geometry.x.values, g.geometry.y.values
cx, cy = PX.mean(), PY.mean()
Mc = np.c_[PX-cx, PY-cy]
u = np.linalg.svd(Mc-Mc.mean(0), full_matrices=False)[2][0]
if u[0] < 0: u = -u
v = np.array([-u[1], u[0]])
def to_sd(x, y):
    dx, dy = x-cx, y-cy
    return dx*u[0]+dy*u[1], dx*v[0]+dy*v[1]
PS, PD = to_sd(PX, PY)
S0, S1 = PS.min()-10, PS.max()+10
AREA = np.ptp(PS)*2*BUF
Hsup = pd.to_numeric(g["Hsup"], errors="coerce").values if "Hsup" in g else \
       np.full(len(g), np.nan)
strat = g["stratum"].astype(str).values if "stratum" in g else \
        np.array(["?"]*len(g))
P(f"surveyed stems            {len(g)}   (RTK, EPSG:2154)")
P(f"hedgerow axis bearing     {np.degrees(np.arctan2(u[0], u[1]))%180:.1f} deg")
P(f"length along the axis     {np.ptp(PS):.1f} m")
P(f"stem departure from line  median {np.median(np.abs(PD)):.2f} m, "
  f"maximum {np.abs(PD).max():.2f} m")
if np.abs(PD).max() > 3.0:
    P("STOP: the stems are not collinear to 3 m, a straight axis is wrong")
    sys.exit(1)
P(f"strip read                {np.ptp(PS):.0f} m long, {2*BUF:.0f} m wide")
A.update(n_stems=int(len(g)), axis_len=float(np.ptp(PS)),
         stem_off_max=float(np.abs(PD).max()))
def clipf(x, y):
    s, d = to_sd(x, y)
    return (np.abs(d) <= BUF)&(s >= S0)&(s <= S1), s, d

# ================================================================ 2. laser
H("2. LASER — IGN LiDAR HD, February, leaf-off")
with laspy.open(str(ALS)) as fh:
    hd = fh.header
    crs = hd.parse_crs()
    P(f"file            {ALS.name}")
    P(f"points in file  {hd.point_count:,}")
    P(f"CRS             {crs.to_string() if crs else 'NOT DECLARED'}")
    P(f"vertical datum  declared by the file, not assumed")
    dimA = set(hd.point_format.dimension_names)
aX=[];aY=[];aZ=[];aC=[];aN=[];aR=[]
with laspy.open(str(ALS)) as fh:
    for p in fh.chunk_iterator(CHUNK):
        x, y = np.asarray(p.x), np.asarray(p.y)
        m, s, d = clipf(x, y)
        if not m.any(): continue
        aX.append(x[m]); aY.append(y[m]); aZ.append(np.asarray(p.z)[m])
        aC.append(np.asarray(p.classification)[m])
        aN.append(np.asarray(p.number_of_returns)[m] if "number_of_returns" in dimA
                  else np.ones(int(m.sum()), np.uint8))
        aR.append(np.asarray(p.return_number)[m] if "return_number" in dimA
                  else np.ones(int(m.sum()), np.uint8))
aX=np.concatenate(aX);aY=np.concatenate(aY);aZ=np.concatenate(aZ)
aC=np.concatenate(aC);aN=np.concatenate(aN);aR=np.concatenate(aR)
aS, aD = to_sd(aX, aY)
P(f"\nread into the strip  {aX.size:,}   {aX.size/AREA:.1f} points per m2")
P(f"\n{'class':>6}{'points':>12}{'share':>9}   description")
for c in np.unique(aC):
    n = int((aC == c).sum())
    P(f"{int(c):>6}{n:>12,}{n/aX.size*100:>8.2f}%   {CLS.get(int(c),'?')}")
P(f"\nsingle-return points   {(aN==1).mean()*100:>5.1f} %")
P(f"multiple-return points {(aN>1).mean()*100:>5.1f} %")
A.update(als_read=int(aX.size), als_dens=float(aX.size/AREA),
         als_multi=float((aN>1).mean()*100))

# ============================================================== 3. terrain
H("3. TERRAIN, FROM THE LASER GROUND CLASS")
gm = aC == 2
inh = np.abs(aD) <= HEDGE
P(f"ground returns                    {int(gm.sum()):,}")
P(f"   beneath the hedge (|d| <= {HEDGE:.0f} m) "
  f"{(gm&inh).sum()/(np.ptp(PS)*2*HEDGE):>8.1f} per m2")
P(f"   over open ground               "
  f"{(gm&~inh).sum()/(np.ptp(PS)*2*(BUF-HEDGE)):>8.1f} per m2")
P("The drone cannot supply this. The canopy closes over most of the hedgerow")
P("and photogrammetry reconstructs only surfaces it can see.")
gs0, gd0 = S0-5, -BUF-5
NSg = int(np.ceil(((S1+5)-gs0)/DRES))+1
NDg = int(np.ceil((2*(BUF+5))/DRES))+1
dtm = np.full((NDg, NSg), np.inf)
np.minimum.at(dtm, (np.clip(((aD[gm]-gd0)/DRES).astype(int), 0, NDg-1),
                    np.clip(((aS[gm]-gs0)/DRES).astype(int), 0, NSg-1)), aZ[gm])
meas = np.isfinite(dtm)&(dtm < np.inf)
P(f"\nterrain grid {NDg} x {NSg} at {DRES:.2f} m")
P(f"cells holding a ground return     {meas.mean()*100:.1f} %")
_, ii = ndimage.distance_transform_edt(~meas, return_indices=True)
dtm = ndimage.gaussian_filter(np.where(meas, dtm, 0.)[tuple(ii)], 1.5).astype("float32")
def terrain(s, d):
    return ndimage.map_coordinates(
        dtm, [np.clip((d-gd0)/DRES, 0, NDg-1), np.clip((s-gs0)/DRES, 0, NSg-1)],
        order=1, mode="nearest").astype("float64")
gres = aZ[gm] - terrain(aS[gm], aD[gm])
rin = np.abs(aD[gm]) <= HEDGE
P(f"\nscatter of ground returns about the fitted surface")
P(f"   beneath the hedge   +/- {gres[rin].std():.3f} m")
P(f"   over open ground    +/- {gres[~rin].std():.3f} m")
P(f"   overall             +/- {gres.std():.3f} m")
P(f"This is the terrain model's own uncertainty. Every canopy height computed")
P(f"in this project is a difference against this surface and carries it.")
relief = np.ptp(terrain(np.array([S0, S1]), np.array([0., 0.])))
P(f"\nrelief along the hedgerow  {relief:.2f} m over {np.ptp(PS):.0f} m")
A.update(ground_hedge=float((gm&inh).sum()/(np.ptp(PS)*2*HEDGE)),
         dtm_cells=float(meas.mean()*100),
         dtm_sd_hedge=float(gres[rin].std()), dtm_sd_field=float(gres[~rin].std()),
         relief=float(relief))

# ================================================================ 4. drone
H("4. DRONE — photogrammetry, May, leaf-on")
with laspy.open(str(UAV)) as fh:
    hd = fh.header
    crs = hd.parse_crs()
    extra = list(hd.point_format.extra_dimension_names)
    HASC = "confidence" in extra
    hasrgb = {"red","green","blue"} <= set(hd.point_format.dimension_names)
    P(f"file            {UAV.name}")
    P(f"points in file  {hd.point_count:,}")
    P(f"CRS             {crs.to_string() if crs else 'NOT DECLARED'}")
    P(f"extra fields    {extra}")
    P(f"confidence      {'present' if HASC else 'ABSENT'}")
    P(f"colour          {'present' if hasrgb else 'absent'}")

cs0, cd0 = S0, -BUF
NCS = int(np.ceil((S1-S0)/CRES))+1
NCD = int(np.ceil(2*BUF/CRES))+1
ztop = np.full((NCD, NCS), -np.inf, "float32")   # -inf and +inf, never nan:
zbot = np.full((NCD, NCS),  np.inf, "float32")   # maximum.at into a nan array
cnt  = np.zeros((NCD, NCS), "int32")             # leaves it nan for ever
conf_all = np.zeros(256, "int64"); conf_hedge = np.zeros(256, "int64")
uc = {}
wc0 = (S0+S1)/2-WINLEN/2; wc1 = wc0+WINLEN
W = [[], [], [], [], []]; T = [[], [], [], [], []]
ntot = 0; t0 = time.time()
with laspy.open(str(UAV)) as fh:
    for p in fh.chunk_iterator(CHUNK):
        x, y = np.asarray(p.x), np.asarray(p.y)
        m, s, d = clipf(x, y)
        if not m.any(): continue
        s, d = s[m], d[m]; z = np.asarray(p.z)[m]
        cl = np.asarray(p.classification)[m]
        for c in np.unique(cl): uc[int(c)] = uc.get(int(c), 0)+int((cl == c).sum())
        cf = (np.asarray(p["confidence"])[m].astype("int32") if HASC
              else np.full(s.size, 255, "int32"))
        cc = np.clip(cf, 0, 255)
        conf_all += np.bincount(cc, minlength=256)
        conf_hedge += np.bincount(cc[np.abs(d) <= HEDGE], minlength=256)
        col = (np.c_[np.asarray(p.red)[m], np.asarray(p.green)[m],
                     np.asarray(p.blue)[m]].astype("float32")/65535.
               if hasrgb else np.zeros((s.size, 3), "float32"))
        ri = np.clip(((d-cd0)/CRES).astype(int), 0, NCD-1)
        ci = np.clip(((s-cs0)/CRES).astype(int), 0, NCS-1)
        np.maximum.at(ztop, (ri, ci), z.astype("float32"))
        np.minimum.at(zbot, (ri, ci), z.astype("float32"))
        np.add.at(cnt, (ri, ci), 1)
        w = (s >= wc0)&(s < wc1)
        if w.any():
            for a_, b_ in zip(W, (s[w], d[w], z[w], cf[w], col[w])): a_.append(b_)
        t = np.arange((-ntot) % STRIDE, s.size, STRIDE)
        if t.size:
            for a_, b_ in zip(T, (s[t], d[t], z[t], cf[t], col[t])): a_.append(b_)
        ntot += s.size
ztop = np.where(ztop > -1e30, ztop, np.nan)
zbot = np.where(zbot <  1e30, zbot, np.nan)
hb = np.abs(cd0+np.arange(NCD)*CRES) <= HEDGE
P(f"\nread into the strip  {ntot:,}   {ntot/AREA:.0f} points per m2  "
  f"({time.time()-t0:.0f} s)")
P(f"\n{'class':>6}{'points':>14}{'share':>9}   description")
for c in sorted(uc):
    P(f"{c:>6}{uc[c]:>14,}{uc[c]/ntot*100:>8.2f}%   {CLS.get(c,'?')}")
if len(uc) <= 1: P("\nNOTE: a single class only — this delivery is not classified")
P(f"\nsurface grid {NCD} x {NCS} at {CRES:.2f} m")
P(f"cells holding a point, over the hedge  "
  f"{np.isfinite(ztop[hb]).mean()*100:.1f} %")
P(f"point density over the hedge           "
  f"{cnt[hb].sum()/(np.ptp(PS)*2*HEDGE):.0f} per m2")
if HASC:
    nz = np.nonzero(conf_all)[0]
    P(f"\nconfidence values present {nz.min()} to {nz.max()}")
    P(f"{'confidence <=':>14}{'% of strip':>12}{'% of hedge':>12}")
    ca, ch = np.cumsum(conf_all), np.cumsum(conf_hedge)
    for t_ in range(0, 7):
        P(f"{t_:>14}{ca[t_]/ntot*100:>11.3f}%"
          f"{ch[t_]/max(conf_hedge.sum(),1)*100:>11.3f}%")
A.update(uav_read=int(ntot), uav_dens=float(ntot/AREA),
         uav_dens_hedge=float(cnt[hb].sum()/(np.ptp(PS)*2*HEDGE)),
         uav_classes={str(k): int(v) for k, v in uc.items()}, has_conf=bool(HASC))

# ================================================= 5. offset, measured only
H("5. VERTICAL OFFSET BETWEEN THE TWO DATASETS — MEASURED, NOT APPLIED")
P("Both surfaces are elevations in their own file's datum. The three estimates")
P("below are reported so that B2 can choose between them on evidence.")
sg, dg = np.meshgrid(cs0+np.arange(NCS)*CRES, cd0+np.arange(NCD)*CRES)
TER = terrain(sg.ravel(), dg.ravel()).reshape(sg.shape)
fld = (np.abs(dg) >= FIELD[0])&(np.abs(dg) <= FIELD[1])&np.isfinite(zbot)
diff = (zbot-TER)[fld]; diff = diff[np.isfinite(diff)]
q = np.percentile(diff, [5, 25, 50, 75, 95])
P(f"\n(A) open ground, {FIELD[0]:.0f} to {FIELD[1]:.0f} m from the hedge, "
  f"{diff.size:,} cells")
P(f"    lowest drone elevation minus terrain")
P(f"    p5 {q[0]:+.3f}  p25 {q[1]:+.3f}  median {q[2]:+.3f}  "
  f"p75 {q[3]:+.3f}  p95 {q[4]:+.3f}")
lm = np.median(diff[(dg[fld] < 0)]); rm = np.median(diff[(dg[fld] > 0)])
P(f"    left side {lm:+.3f}   right side {rm:+.3f}   "
  f"difference {abs(lm-rm):.3f} m")
P(f"\n(B) the 5th percentile of the same distribution: {q[0]:+.3f} m")
P( "    the argument being that ruts and bare patches let the drone reach")
P( "    real soil in a minority of cells")

hi = (strat == "high_stand")&np.isfinite(Hsup)&(Hsup > 4)
kt = cKDTree(np.c_[PS, PD])
exposed = np.zeros(len(PS), bool)
for k in np.where(hi)[0]:
    nb = [j for j in kt.query_ball_point([PS[k], PD[k]], NEIGH) if j != k]
    exposed[k] = not any(np.isfinite(Hsup[j]) and Hsup[j] > Hsup[k] for j in nb)
raw = (ztop-TER)
kg = cKDTree(np.c_[sg.ravel(), dg.ravel()])
flat = raw.ravel()
top = np.full(len(PS), np.nan)
for k in np.where(exposed)[0]:
    val = flat[kg.query_ball_point([PS[k], PD[k]], RSTEM)]
    val = val[np.isfinite(val)]
    if val.size: top[k] = val.max()
mm = exposed & np.isfinite(top)
P(f"\n(C) surveyed trees whose own crown top is visible from above")
P(f"    high stand with a measured height  {int(hi.sum())}")
P(f"    of which no taller stem within {NEIGH:.0f} m  {int(exposed.sum())}")
P(f"    with drone material within {RSTEM:.2f} m    {int(mm.sum())}")
if mm.sum() >= 12:
    ts = stats.theilslopes(top[mm], Hsup[mm], 0.95)
    P(f"    Theil-Sen regression of drone elevation above terrain on measured "
      f"height")
    P(f"       slope     {ts[0]:.3f}   95 % interval {ts[2]:.3f} to {ts[3]:.3f}")
    P(f"       intercept {ts[1]:+.3f} m")
    P(f"       median difference (drone minus measured) "
      f"{np.median(top[mm]-Hsup[mm]):+.3f} m")
    if ts[2] <= 1.0 <= ts[3]:
        P(f"    the slope is consistent with one, so a constant shift is an")
        P(f"    adequate model and the offset is the intercept")
    else:
        P(f"    the slope is NOT consistent with one, so the two are not")
        P(f"    related by a constant shift and no single offset can be right")
    A.update(ts_slope=float(ts[0]), ts_lo=float(ts[2]), ts_hi=float(ts[3]),
             ts_intercept=float(ts[1]),
             med_diff=float(np.median(top[mm]-Hsup[mm])), n_exposed=int(mm.sum()))
else:
    P("    too few usable trees; estimate (C) cannot be formed")
A.update(off_field_median=float(q[2]), off_field_p5=float(q[0]),
         off_side_diff=float(abs(lm-rm)))
P("\nNo offset has been applied. Every array written below is an elevation.")

# ================================================================ 6. write
H("6. WRITE")
cat = lambda a: np.concatenate(a) if a else np.zeros(0)
vat = lambda a: np.vstack(a) if a else np.zeros((0, 3), "float32")
wS,wD,wZ,wF,wC = cat(W[0]),cat(W[1]),cat(W[2]),cat(W[3]),vat(W[4])
if wS.size > WINCAP:
    k = np.random.default_rng(0).choice(wS.size, WINCAP, replace=False)
    wS,wD,wZ,wF,wC = wS[k],wD[k],wZ[k],wF[k],wC[k]
np.savez_compressed(
    OUT/"ennezat_B1.npz",
    axis=np.array([cx, cy, u[0], u[1]]), srange=np.array([S0, S1]),
    buf=np.array([BUF]), hedge=np.array([HEDGE]), field=np.array(FIELD),
    win=np.array([wc0, wc1]),
    dtm=dtm, dtm_org=np.array([gs0, gd0, DRES]),
    terrain_on_grid=TER.astype("float32"),
    uav_ztop=ztop.astype("float32"), uav_zbot=zbot.astype("float32"),
    uav_count=cnt, grid_org=np.array([cs0, cd0, CRES]),
    conf_all=conf_all, conf_hedge=conf_hedge,
    als_s=aS, als_d=aD, als_z=aZ, als_c=aC, als_nret=aN, als_rnum=aR,
    als_gres=gres, als_gres_d=aD[gm],
    win_s=wS, win_d=wD, win_z=wZ, win_conf=wF, win_rgb=wC,
    thin_s=cat(T[0]), thin_d=cat(T[1]), thin_z=cat(T[2]),
    thin_conf=cat(T[3]), thin_rgb=vat(T[4]),
    stem_s=PS, stem_d=PD, stem_x=PX, stem_y=PY,
    stem_exposed=exposed, stem_topraw=top,
    audit=np.array([json.dumps(A)], dtype=object),
    **{f"stem_{c}": g[c].astype(str).values for c in
       ("variety","dominance","stratum","state","trunkNumber","Hsup","Hinf",
        "HLCE","gpsID") if c in g.columns})
P(f"ennezat_B1.npz   {(OUT/'ennezat_B1.npz').stat().st_size/1e6:.0f} MB")
P("contents: elevations, terrain, audit. NO heights, NO offset applied.")
(OUT/"REPORT_B1.md").write_text(
    "# Stage B1 — Reading and terrain\n\n```\n"+"\n".join(L)+"\n```\n",
    encoding="utf-8")
(OUT/"ennezat_B1.json").write_text(json.dumps(A, indent=2))
P(f"\ntotal {time.time()-t_all:.0f} s")
P("\nNext: python preparation/02_vertical_offset_road.py")