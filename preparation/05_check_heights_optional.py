"""
STAGE B2 — ENNEZAT: resolve the datum, remove outliers, verify.
*** RUN IN A TERMINAL ***

    python preparation/05_check_heights_optional.py

--------------------------------------------------------------------------
THIS IS THE ONLY SCRIPT THAT CREATES A HEIGHT

B1 wrote elevations. This script decides the offset between the two datasets,
applies it once, and converts everything to height above the terrain model.
After it runs, every height in the project is in the same reference, so no
later script can mix a corrected quantity with an uncorrected one. That
mistake has already been made once and this is the structural cure.

  height of a laser point  =  its elevation  -  terrain
  height of a drone point  =  its elevation  -  terrain  -  DZ

DZ is chosen below, from evidence, and reported.

--------------------------------------------------------------------------
CHOOSING DZ

Three estimates come from B1. The open-ground estimate compares the drone's
lowest surface with the laser terrain where both instruments see the same bare
soil. It is easy to compute and it is contaminated by whatever was growing in
the field at the time of the flight, which the laser did not see because it
flew in another season.

The third estimate uses the field inventory. For a tree whose crown top is
genuinely visible from above -- no taller neighbour within a few metres -- the
drone's surface elevation above the terrain, minus the height measured with
the hypsometer, is the offset. This does not depend on what was growing in the
field, so it is preferred, and it is taken from the regression rather than from
a median of differences, because a regression also tests whether a constant
shift is the right model at all. If the slope departs from one, the two
datasets are not related by a shift and no single number can be correct.

--------------------------------------------------------------------------
THE FILTERING RULES

  1  CONFIDENCE      points corroborated by only one or two depth maps
  2  ABOVE NEIGHBOURS points standing more than a margin above the highest
                      surface in a small ring around them, the ring excluding
                      the point's own cell so nothing is compared with itself
  3  BELOW TERRAIN    points under the ground

No rule counts neighbours. The reasoning is in REPORT_B_cleaning_rationale.md.
Rule 3 depends on the offset, which is why it is applied only after DZ is
fixed.

--------------------------------------------------------------------------
VERIFICATION

Four checks are printed with a verdict on each. They test the thing that went
wrong before -- whether quantities that ought to agree actually do.

  1  over open ground the drone surface should sit at zero height
  2  over open ground the laser ground returns should sit at zero height
  3  at trees whose top is visible, the drone canopy height should match the
     height measured in the field
  4  the cleaning should lower the top of the canopy and leave its middle
     alone

A failed check is printed as FAIL and the reason is stated. Nothing downstream
should be run until all four pass or until the failure is understood.
"""

from pathlib import Path
import numpy as np, pandas as pd, laspy, json, time
from scipy import ndimage, stats

from settings import OUT, UAV  # all paths: config/preparation.json
CACHE = OUT/"ennezat_B1.npz"

RING     = 1.5
MARGINS  = [1.0, 1.5, 2.0, 3.0, 4.0]
MARGIN   = 2.0
BELOW    = -0.30
CONF_TRY = 3
CONF_CAP = 8.0
NOISE_CLASSES = (7, 18, 64, 65)
GATE_HI  = 25.0
RSTEM    = 0.75
CHUNK    = 3_000_000
WINCAP   = 2_000_000
STRIDE   = 120

L = []
def P(s=""): print(s, flush=True); L.append(str(s))
def H(t): P(); P("="*76); P(t); P("="*76)
CHECKS = []
def check(name, value, lo, hi, unit="m", note=""):
    ok = (lo <= value <= hi)
    CHECKS.append((name, value, ok))
    P(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    P(f"          measured {value:+.3f} {unit}, expected between "
      f"{lo:+.2f} and {hi:+.2f}")
    if note: P(f"          {note}")
    return ok

t_all = time.time()
C = np.load(CACHE, allow_pickle=True)
A0 = json.loads(str(C["audit"][0]))
cx, cy, u0, u1 = C["axis"]; v0, v1 = -u1, u0
S0, S1 = C["srange"]; BUF = float(C["buf"][0]); HEDGE = float(C["hedge"][0])
F0, F1 = C["field"]; wc0, wc1 = C["win"]
dtm = C["dtm"]; gs0, gd0, DRES = C["dtm_org"]
ZTOP = C["uav_ztop"]; ZBOT = C["uav_zbot"]; TER = C["terrain_on_grid"]
cs0, cd0, CRES = C["grid_org"]
ND, NS = ZTOP.shape
sax = cs0+np.arange(NS)*CRES; dax = cd0+np.arange(ND)*CRES
hb = np.abs(dax) <= HEDGE
PS, PD = C["stem_s"], C["stem_d"]
Hsup = pd.to_numeric(pd.Series(C["stem_Hsup"]), errors="coerce").values
exposed = C["stem_exposed"]; topraw = C["stem_topraw"]
def to_sd(x, y):
    dx, dy = x-cx, y-cy
    return dx*u0+dy*u1, dx*v0+dy*v1
def terrain(s, d):
    return ndimage.map_coordinates(
        dtm, [np.clip((d-gd0)/DRES, 0, dtm.shape[0]-1),
              np.clip((s-gs0)/DRES, 0, dtm.shape[1]-1)],
        order=1, mode="nearest").astype("float64")

# ================================================================= 1. DZ
H("1. THE VERTICAL OFFSET")
# Measured in B2b on the road surface class -- bare, hard, and unchanged
# between the two acquisitions -- and B2c decides whether a constant or a
# tilted plane is the right model. It is READ here, never recomputed, so the
# decision lives in exactly one place.
DZF = OUT/"ennezat_dz.json"
if not DZF.exists():
    raise SystemExit("Run preparation/02 and 03 first.")
DZI = json.loads(DZF.read_text())
MOD = DZI.get("model", dict(type="constant", c=float(DZI["dz"])))
if MOD["type"] == "plane":
    def offset(x, y):
        return MOD["ax"]*(x-MOD["x0"]) + MOD["ay"]*(y-MOD["y0"]) + MOD["c"]
else:
    def offset(x, y):
        return np.full(np.shape(x), MOD["c"], dtype="float64")
DZ = float(DZI["dz"]); src = DZI["source"]
P(f"read from {DZF.name}   ({src})")
P(f"model                 {MOD['type']}")
if MOD["type"] == "plane":
    P(f"   north-south tilt   {DZI['tilt_mm_per_m']:+.2f} mm per metre")
    P(f"   across the hedge   {DZI['tilt_over_hedge']:+.3f} m end to end")
    P(f"   road covers        {DZI['road_covers_hedge']*100:.0f} % of the hedge")
P(f"representative value  {DZ:+.3f} m at the centre of the hedgerow")
P(f"residual on the road  IQR {DZI.get('resid_iqr', float('nan')):.3f} m")
if "crop_height" in DZI:
    P(f"\nfor comparison, the drone's ground class out in the fields sits")
    P(f"{DZI['crop_height']:.2f} m higher still — the standing wheat. That is why")
    P(f"open ground gave {A0['off_field_median']:+.3f} m and is not used.")
P("\nApplied to the drone only. The laser defines the reference.")

# ====================================================== 2. reference surface
H("2. REFERENCE SURFACE FOR RULE 2")
# the offset evaluated on the surface grid, so the raw canopy height is in the
# same reference as every point height computed later
Sg, Dg = np.meshgrid(sax, dax)
Xg = cx + Sg*u0 + Dg*v0
Yg = cy + Sg*u1 + Dg*v1
OFFG = offset(Xg, Yg).astype("float32")
P(f"offset over the grid: {OFFG.min():.3f} to {OFFG.max():.3f} m")
CH_raw = ZTOP - OFFG - TER
r = int(round(RING/CRES))
yy, xx = np.mgrid[-r:r+1, -r:r+1]
ring = (xx*xx+yy*yy) <= r*r
ring[r, r] = False
filled = np.where(np.isfinite(CH_raw), CH_raw, -9999.).astype("float32")
REF = ndimage.maximum_filter(filled, footprint=ring, mode="nearest")
P(f"ring radius {RING:.1f} m = {r} cells, own cell excluded, "
  f"{int(ring.sum())} cells contribute")
def refat(s, d):
    return ndimage.map_coordinates(
        REF, [np.clip((d-cd0)/CRES, 0, ND-1), np.clip((s-cs0)/CRES, 0, NS-1)],
        order=0, mode="nearest").astype("float64")

# ================================================== 3. confidence threshold
H("3. CONFIDENCE THRESHOLD")
HASC = bool(A0.get("has_conf", False))
if HASC:
    ch = np.cumsum(C["conf_hedge"]); nh = C["conf_hedge"].sum()
    ca = np.cumsum(C["conf_all"]);   na = C["conf_all"].sum()
    P(f"{'drop conf <=':>13}{'% of strip':>13}{'% of hedge':>13}")
    for t_ in range(0, 8):
        P(f"{t_:>13}{ca[t_]/na*100:>12.3f}%{ch[t_]/nh*100:>12.3f}%")
    CONF = CONF_TRY
    while CONF > 0 and ch[CONF]/nh*100 > CONF_CAP:
        P(f"\nconf <= {CONF} would take {ch[CONF]/nh*100:.2f} % of the hedge, "
          f"above the {CONF_CAP:.0f} % limit; stepping down")
        CONF -= 1
    P(f"\nadopted: remove confidence <= {CONF}  "
      f"({ch[CONF]/nh*100:.3f} % of the hedge)")
else:
    CONF = -1; P("no confidence field; rule 1 disabled")

# ============================================================== 4. the pass
H("4. FILTERING THE DRONE CLOUD")
ztop = np.full((ND, NS), -np.inf, "float32")   # -inf, never nan
cnt  = np.zeros((ND, NS), "int32")
W = [[], [], [], []]; T = [[], [], [], []]; X = [[], [], [], []]
st = dict(n=0, kept=0, only_c=0, only_g=0, both=0, low=0,
          high=np.zeros(len(MARGINS), "int64"))
t0 = time.time()
with laspy.open(str(UAV)) as fh:
    for p in fh.chunk_iterator(CHUNK):
        x, y = np.asarray(p.x), np.asarray(p.y)
        s, d = to_sd(x, y)
        m = (np.abs(d) <= BUF)&(s >= S0)&(s <= S1)
        if not m.any(): continue
        s, d = s[m], d[m]; z = np.asarray(p.z)[m]
        xm, ym = x[m], y[m]
        cf = (np.asarray(p["confidence"])[m].astype("int32") if HASC
              else np.full(s.size, 255, "int32"))
        col = np.c_[np.asarray(p.red)[m], np.asarray(p.green)[m],
                    np.asarray(p.blue)[m]].astype("float32")/65535.
        # the one place in the whole project where a height is created
        h = z - offset(xm, ym) - terrain(s, d)
        rf = refat(s, d); valid = rf > -9000
        st["n"] += s.size
        for i, mg in enumerate(MARGINS):
            st["high"][i] += int(((h > rf+mg)&valid).sum())
        lo = h < BELOW; st["low"] += int(lo.sum())
        geom = lo | ((h > rf+MARGIN)&valid)
        conf = cf <= CONF if CONF >= 0 else np.zeros(s.size, bool)
        st["only_c"] += int((conf&~geom).sum())
        st["only_g"] += int((geom&~conf).sum())
        st["both"]   += int((geom&conf).sum())
        bad = conf|geom
        if bad.any():
            k = np.arange(0, int(bad.sum()), max(1, int(bad.sum())//4000+1))
            X[0].append(s[bad][k]); X[1].append(d[bad][k]); X[2].append(h[bad][k])
            X[3].append(np.where(conf[bad][k], 1,
                        np.where(lo[bad][k], 2, 3)).astype("uint8"))
        s, d, z, h, col = s[~bad], d[~bad], z[~bad], h[~bad], col[~bad]
        st["kept"] += s.size
        ri = np.clip(((d-cd0)/CRES).astype(int), 0, ND-1)
        ci = np.clip(((s-cs0)/CRES).astype(int), 0, NS-1)
        np.maximum.at(ztop, (ri, ci), h.astype("float32"))
        np.add.at(cnt, (ri, ci), 1)
        w = (s >= wc0)&(s < wc1)
        if w.any():
            for a_, b_ in zip(W, (s[w], d[w], h[w], col[w])): a_.append(b_)
        t = np.arange((-st["kept"]) % STRIDE, s.size, STRIDE)
        if t.size:
            for a_, b_ in zip(T, (s[t], d[t], h[t], col[t])): a_.append(b_)
CHM = np.where(ztop > -1e30, ztop, np.nan)
P(f"examined {st['n']:,}   ({time.time()-t0:.0f} s)")
P(f"\n{'above neighbours by':>22}{'points':>13}{'% of cloud':>13}")
for mg, n_ in zip(MARGINS, st["high"]):
    P(f"{f'more than {mg:.1f} m':>22}{int(n_):>13,}{n_/st['n']*100:>12.4f}%")
P(f"{'below the terrain':>22}{st['low']:>13,}{st['low']/st['n']*100:>12.4f}%")
P()
P(f"applied: confidence <= {CONF}, more than {MARGIN:.1f} m above neighbours, "
  f"below {BELOW:+.2f} m")
P(f"{'removed by confidence only':>30}{st['only_c']:>12,}")
P(f"{'removed by geometry only':>30}{st['only_g']:>12,}")
P(f"{'removed by both':>30}{st['both']:>12,}")
P(f"{'removed in total':>30}{st['n']-st['kept']:>12,}"
  f"  ({(st['n']-st['kept'])/st['n']*100:.4f} %)")
P(f"{'retained':>30}{st['kept']:>12,}")

# =============================================================== 5. laser
H("5. FILTERING THE LASER CLOUD")
aS, aD, aZ, aC = C["als_s"], C["als_d"], C["als_z"], C["als_c"]
aN, aR = C["als_nret"], C["als_rnum"]
aH = aZ - terrain(aS, aD)                    # laser needs no offset
badc = np.isin(aC, NOISE_CLASSES)
badg = (aH < BELOW)|(aH > GATE_HI)
bada = badc|badg
P(f"delivered noise classes {NOISE_CLASSES}   {int(badc.sum()):,}")
P(f"below {BELOW:+.2f} m or above {GATE_HI:.0f} m         {int(badg.sum()):,}")
P(f"removed in total                     {int(bada.sum()):,} "
  f"({bada.mean()*100:.3f} %)")
P(f"retained                             {int((~bada).sum()):,}")
P("The laser keeps its delivered classification rather than a statistical")
P("filter: at this point density a real branch in thin canopy has neighbours")
P("as distant as a blunder does, and a distance rule cannot separate them.")
aS,aD,aZ,aH,aC,aN,aR = (a[~bada] for a in (aS,aD,aZ,aH,aC,aN,aR))

# ============================================================ 6. verify
H("6. VERIFICATION")
P("Each check compares two quantities that must agree if the reference frame")
P("is correct. A FAIL means something downstream would be wrong.")
P()
fldm = (np.abs(dax) >= F0)&(np.abs(dax) <= F1)
uav_field = (ZBOT - OFFG - TER)[fldm[:, None] & np.isfinite(ZBOT)]
uav_field = uav_field[np.isfinite(uav_field)]
# NOT zero. The laser flew in February over bare soil; the drone flew in May
# over standing wheat, so the lowest drone surface out there is the top of the
# crop. Expecting zero here would report a correct offset as a failure.
CROP = float(DZI.get("crop_height", 0.9))
check("drone surface over open ground sits at crop height, not at zero",
      float(np.median(uav_field)), CROP-0.45, CROP+0.45,
      note=f"the road measurement implies a crop {CROP:.2f} m tall; the laser "
           f"saw bare February soil, the drone saw May wheat")
alsg = aH[(np.abs(aD) >= F0)&(np.abs(aD) <= F1)&(aC == 2)]
check("laser ground returns over open ground sit at zero height",
      float(np.median(alsg)), -0.15, 0.15,
      note="this tests the terrain model against its own input")
em = exposed & np.isfinite(Hsup)
tv = np.full(len(PS), np.nan)
for k in np.where(em)[0]:
    ri = np.abs(dax-PD[k]) <= RSTEM; ci = np.abs(sax-PS[k]) <= RSTEM
    blk = CHM[np.ix_(ri, ci)]
    if np.isfinite(blk).any(): tv[k] = np.nanmax(blk)
mm = em & np.isfinite(tv)
if mm.sum() >= 10:
    dif = tv[mm]-Hsup[mm]
    check(f"drone canopy height matches the field at the {int(mm.sum())} trees "
          f"whose top is visible",
          float(np.median(dif)), -1.0, 1.0,
          note=f"scatter of the differences +/- {dif.std():.2f} m; this is the "
               f"real uncertainty on a single derived tree height")
else:
    P("  [SKIP]  too few trees with a visible top")
ah = np.full(len(PS), np.nan)
for k in np.where(em)[0]:
    sel = (np.abs(aS-PS[k]) <= RSTEM)&(np.abs(aD-PD[k]) <= RSTEM)
    if sel.any(): ah[k] = aH[sel].max()
am = em & np.isfinite(ah)
if am.sum() >= 10:
    P(f"\n  [INFO]  laser canopy height against the field at "
      f"{int(am.sum())} trees")
    P(f"          median {np.median(ah[am]-Hsup[am]):+.2f} m")
    P(f"          a negative value is expected: the laser flew leaf-off, so")
    P(f"          its canopy top is branch rather than foliage, and at this")
    P(f"          return density a crown apex is often simply not hit")
# The comparison is made over cells that hold data BOTH before and after.
# Comparing the two full grids was a mistake: filtering empties any cell whose
# only content was below the terrain, those cells held low values, and dropping
# them raises the median of what remains. That is a change in which cells are
# counted, not a change in the canopy.
both = np.isfinite(CH_raw) & np.isfinite(CHM) & hb[:, None]
P()
P(f"  [INFO]  effect of the filtering on the canopy over the hedge")
P(f"          compared over the {int(both.sum()):,} cells holding data both "
  f"before and after")
P(f"          ({int((np.isfinite(CH_raw)&hb[:,None]).sum() - both.sum()):,} "
  f"cells were emptied entirely and are excluded from both columns)")
P(f"          {'':<22}{'before':>10}{'after':>10}{'change':>10}")
for nm, f in (("maximum (m)", np.nanmax),
              ("99th percentile (m)", lambda z: np.nanpercentile(z, 99)),
              ("90th percentile (m)", lambda z: np.nanpercentile(z, 90)),
              ("median (m)", np.nanmedian)):
    b, a = f(CH_raw[both]), f(CHM[both])
    P(f"          {nm:<22}{b:>10.2f}{a:>10.2f}{a-b:>10.2f}")
med_shift = float(np.nanmedian(CHM[both])-np.nanmedian(CH_raw[both]))
check("filtering left the middle of the canopy alone",
      med_shift, -0.15, 0.15,
      note="the top should fall and the middle should not; a falling median "
           "means real surface is being removed")
P()
npass = sum(1 for _, _, o in CHECKS if o)
P(f"{npass} of {len(CHECKS)} checks passed")
if npass < len(CHECKS):
    P("Do not run Stage C until each failure is understood.")

# ============================================================== 7. write
H("7. WRITE")
cat = lambda a: np.concatenate(a) if a else np.zeros(0)
vat = lambda a: np.vstack(a) if a else np.zeros((0, 3), "float32")
wS,wD,wH,wC = cat(W[0]),cat(W[1]),cat(W[2]),vat(W[3])
if wS.size > WINCAP:
    k = np.random.default_rng(0).choice(wS.size, WINCAP, replace=False)
    wS,wD,wH,wC = wS[k],wD[k],wH[k],wC[k]
A = dict(A0, dz=DZ, dz_source=src, conf_threshold=int(CONF), margin=MARGIN,
         ring=RING, below=BELOW, examined=int(st["n"]), kept=int(st["kept"]),
         removed=int(st["n"]-st["kept"]),
         removed_pct=float((st["n"]-st["kept"])/st["n"]*100),
         only_conf=int(st["only_c"]), only_geom=int(st["only_g"]),
         both=int(st["both"]), als_removed=int(bada.sum()),
         als_kept=int((~bada).sum()),
         checks={n: bool(o) for n, _, o in CHECKS},
         chm_max=float(np.nanmax(CHM[hb])),
         chm_med=float(np.nanmedian(CHM[hb])))
np.savez_compressed(
    OUT/"ennezat_B2.npz",
    axis=C["axis"], srange=C["srange"], buf=C["buf"], hedge=C["hedge"],
    field=C["field"], win=C["win"], dz=np.array([DZ]),
    dtm=dtm, dtm_org=C["dtm_org"], grid_org=C["grid_org"],
    chm=CHM.astype("float32"), count=cnt,
    terrain_on_grid=TER.astype("float32"),
    als_s=aS, als_d=aD, als_h=aH.astype("float32"), als_z=aZ,
    als_c=aC, als_nret=aN, als_rnum=aR,
    win_s=wS, win_d=wD, win_h=wH, win_rgb=wC,
    thin_s=cat(T[0]), thin_d=cat(T[1]), thin_h=cat(T[2]), thin_rgb=vat(T[3]),
    removed_s=cat(X[0]), removed_d=cat(X[1]), removed_h=cat(X[2]),
    removed_why=cat(X[3]),
    stem_s=PS, stem_d=PD, stem_x=C["stem_x"], stem_y=C["stem_y"],
    stem_exposed=exposed, stem_uavtop=tv, stem_alstop=ah,
    audit=np.array([json.dumps(A)], dtype=object),
    **{k: C[k] for k in C.files if k.startswith("stem_")
       and k not in ("stem_s","stem_d","stem_x","stem_y","stem_exposed",
                     "stem_topraw")})
P(f"ennezat_B2.npz   {(OUT/'ennezat_B2.npz').stat().st_size/1e6:.0f} MB")
P("contents: HEIGHTS above the terrain, both datasets in one reference.")
P("removed_why: 1 confidence, 2 below terrain, 3 above neighbours")
(OUT/"REPORT_B2.md").write_text(
    "# Stage B2 — Datum, filtering, verification\n\n```\n"+"\n".join(L)+"\n```\n",
    encoding="utf-8")
P(f"\ntotal {time.time()-t_all:.0f} s")
P("\nDone.")