"""
STAGE B2c — ENNEZAT: constant offset, or a tilted one?   python preparation/03_offset_constant_or_tilted.py

Reads ennezat_road.npz and ennezat_B1.npz. Seconds, no point cloud.

--------------------------------------------------------------------------
THE QUESTION

The road measurement gave a median offset of about 1.32 m, but also a
north-south gradient of roughly 1.5 mm per metre, which is 0.40 m across the
survey area. The hedgerow spans about 160 m in that direction, so a single
constant would leave a systematic drift of around a quarter of a metre from
one end of the hedge to the other. That is smaller than the tree-to-tree
scatter, but it is systematic rather than random: every tree at one end biased
one way, every tree at the other end the opposite way. It would appear as a
gradient in canopy height along the hedgerow, and would eventually be
interpreted as something real.

Correcting it is a plane instead of a number. The only reason not to is
extrapolation: if the road lies in one corner of the site, a plane fitted
there and extended over the rest is worse than no correction at all.

--------------------------------------------------------------------------
SO THIS DECIDES ON COVERAGE, NOT ON PREFERENCE

  1  where the road points are, mapped, with the hedgerow drawn on top
  2  whether they BRACKET the hedgerow in the tilted direction, or whether
     using them over the hedge means extrapolating
  3  how much the residual scatter actually improves when the tilt is removed
  4  a decision, with the reason stated

A plane is adopted only if the road brackets the hedgerow and the residual
improves. Otherwise the constant stands and the tilt is carried as a stated
uncertainty rather than a silent one.
"""

from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, json
from matplotlib.lines import Line2D

from settings import OUT  # all paths: config/preparation.json
R   = np.load(OUT/"ennezat_road.npz")
B   = np.load(OUT/"ennezat_B1.npz", allow_pickle=True)
DZI = json.loads((OUT/"ennezat_dz.json").read_text())

MARGIN_OK = 0.90   # road must bracket at least this share of the hedge extent
GAIN_OK   = 0.03   # m, minimum improvement in residual scatter to bother

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 250, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 1., "legend.edgecolor": "0.75",
    "figure.constrained_layout.use": True,
    "axes.grid": True, "grid.alpha": .18})
BLUE, RED, GREY = "#1F5FA9", "#B0113B", "#777777"
L = []
def P(s=""): print(s, flush=True); L.append(str(s))
def H(t): P(); P("="*74); P(t); P("="*74)

rx, ry, rd = R["road_x"], R["road_y"], R["road_d"]
cx, cy, u0, u1 = B["axis"]; v0, v1 = -u1, u0
PS = B["stem_s"]
hx = cx + PS*u0; hy = cy + PS*u1          # the hedgerow, in map coordinates

H("1. WHERE THE ROAD IS")
P(f"road points                {rx.size:,}")
P(f"road easting  {rx.min():.0f} to {rx.max():.0f} m   span {np.ptp(rx):.0f} m")
P(f"road northing {ry.min():.0f} to {ry.max():.0f} m   span {np.ptp(ry):.0f} m")
P(f"hedge easting  {hx.min():.0f} to {hx.max():.0f} m   span {np.ptp(hx):.0f} m")
P(f"hedge northing {hy.min():.0f} to {hy.max():.0f} m   span {np.ptp(hy):.0f} m")

H("2. DOES THE ROAD BRACKET THE HEDGE?")
P("The tilt is north-south, so the northing extent is what matters. If the")
P("road does not span the hedge, using a plane over the hedge extrapolates.")
lo = max(ry.min(), hy.min()); hi = min(ry.max(), hy.max())
cover = max(0.0, hi-lo)/np.ptp(hy)
P(f"\nhedge northing covered by road points  {cover*100:.1f} %")
NB = 24
edges = np.linspace(hy.min(), hy.max(), NB+1)
cnt, _ = np.histogram(ry, bins=edges)
empty = int((cnt == 0).sum())
P(f"hedge split into {NB} bands along its northing extent")
P(f"bands with no road point at all        {empty} of {NB}")
if empty:
    gaps = edges[:-1][cnt == 0]
    P(f"widest run of empty bands             "
      f"{np.max(np.diff(np.r_[-1, np.where(cnt>0)[0], NB]))-1} bands "
      f"= {(np.max(np.diff(np.r_[-1, np.where(cnt>0)[0], NB]))-1)*np.ptp(hy)/NB:.0f} m")
bracket = (cover >= MARGIN_OK) and (empty <= NB//4)
P(f"\nverdict: the road {'DOES' if bracket else 'does NOT'} adequately bracket "
  f"the hedgerow")

H("3. HOW MUCH DOES REMOVING THE TILT ACTUALLY HELP?")
const = float(np.median(rd))
A = np.c_[rx-rx.mean(), ry-ry.mean(), np.ones(rx.size)]
coef, *_ = np.linalg.lstsq(A, rd, rcond=None)
fit = A @ coef
res_c = rd - const
res_p = rd - fit
def rob(a): return float(np.percentile(a, 75)-np.percentile(a, 25))
P(f"{'':<34}{'median':>10}{'IQR':>10}{'sd':>10}")
P(f"{'residual after a constant':<34}{np.median(res_c):>+10.3f}"
  f"{rob(res_c):>10.3f}{res_c.std():>10.3f}")
P(f"{'residual after a plane':<34}{np.median(res_p):>+10.3f}"
  f"{rob(res_p):>10.3f}{res_p.std():>10.3f}")
gain = rob(res_c)-rob(res_p)
P(f"\nimprovement in interquartile range     {gain:+.3f} m")
P(f"north-south gradient                   {coef[1]*1000:+.2f} mm per metre")
P(f"across the hedgerow's northing extent  "
  f"{coef[1]*np.ptp(hy):+.3f} m end to end")

H("4. DECISION")
use_plane = bracket and (gain >= GAIN_OK)
if use_plane:
    P("ADOPTED: a tilted correction, fitted on the road.")
    P(f"   DZ(x,y) = {coef[0]:.6e}*(x-{rx.mean():.1f}) "
      f"+ {coef[1]:.6e}*(y-{ry.mean():.1f}) + {coef[2]:.4f}")
    P(f"   over the hedgerow it runs from "
      f"{coef[0]*(hx.min()-rx.mean())+coef[1]*(hy.min()-ry.mean())+coef[2]:.3f} m "
      f"to {coef[0]*(hx.max()-rx.mean())+coef[1]*(hy.max()-ry.mean())+coef[2]:.3f} m")
    P("   The road brackets the hedgerow, so this interpolates rather than")
    P("   extrapolates, and it removes a systematic gradient that a constant")
    P("   would leave in every tree height along the hedge.")
    model = dict(type="plane", ax=float(coef[0]), ay=float(coef[1]),
                 c=float(coef[2]), x0=float(rx.mean()), y0=float(ry.mean()))
    DZrep = float(coef[0]*(hx.mean()-rx.mean())+coef[1]*(hy.mean()-ry.mean())+coef[2])
else:
    if not bracket:
        P("ADOPTED: a constant. The road does not bracket the hedgerow, so a")
        P("plane fitted on it would be extrapolated over the trees, which is")
        P("worse than leaving the tilt in place.")
    else:
        P(f"ADOPTED: a constant. Removing the tilt improves the residual by only")
        P(f"{gain:.3f} m, which does not justify a more complicated model.")
    P(f"   DZ = {const:+.3f} m everywhere")
    P(f"   The tilt of {coef[1]*np.ptp(hy):+.3f} m across the hedgerow is NOT")
    P(f"   corrected and is carried as a stated systematic uncertainty.")
    model = dict(type="constant", c=float(const))
    DZrep = const
P(f"\nrepresentative value at the centre of the hedgerow  {DZrep:+.3f} m")

# ----------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
ax = axs[0]
k = np.random.default_rng(0).choice(rx.size, min(60_000, rx.size), replace=False)
sc = ax.scatter(rx[k], ry[k], c=rd[k], s=1.4, cmap="coolwarm",
                vmin=np.percentile(rd, 2), vmax=np.percentile(rd, 98), lw=0)
ax.plot(hx, hy, "-", color="k", lw=2.4, solid_capstyle="butt")
ax.set_aspect("equal"); ax.grid(False)
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
ax.set_title("(a)", loc="left")
fig.colorbar(sc, ax=ax, shrink=.85, pad=.02, label="Drone minus laser (m)")
ax = axs[1]
bins = np.linspace(ry.min(), ry.max(), 40)
mid = (bins[:-1]+bins[1:])/2
med = np.array([np.median(rd[(ry >= a)&(ry < b)]) if ((ry >= a)&(ry < b)).sum() > 30
                else np.nan for a, b in zip(bins[:-1], bins[1:])])
ax.plot(mid, med, "o", color=BLUE, ms=5, label="Road, binned median")
yy = np.linspace(ry.min(), ry.max(), 50)
ax.plot(yy, coef[1]*(yy-ry.mean())+coef[2], "-", color=RED, lw=1.6,
        label=f"Fitted plane, {coef[1]*1000:+.2f} mm m$^{{-1}}$")
ax.axhline(const, color=GREY, ls="--", lw=1.3, label=f"Constant, {const:.3f} m")
ax.axvspan(hy.min(), hy.max(), color="k", alpha=.06, lw=0)
ax.set_xlabel("Northing (m)"); ax.set_ylabel("Drone minus laser (m)")
ax.legend(loc="upper left")
ax.set_title("(b)", loc="left")
ax = axs[2]
b2 = np.linspace(np.percentile(rd, 0.5)-const, np.percentile(rd, 99.5)-const, 60)
ax.hist(res_c, bins=b2, histtype="step", lw=1.6, color=GREY,
        label=f"After a constant, IQR {rob(res_c):.3f} m")
ax.hist(res_p, bins=b2, histtype="step", lw=1.6, color=RED,
        label=f"After a plane, IQR {rob(res_p):.3f} m")
ax.set_xlabel("Residual (m)"); ax.set_ylabel("Road points")
ax.legend(loc="upper right")
ax.set_title("(c)", loc="left")
fig.legend(handles=[Line2D([], [], color="k", lw=2.4, label="Hedgerow axis")],
           loc="outside upper right")
fig.suptitle("Figure B2c.  Vertical offset between the two acquisitions, "
             "measured on road surface", x=.005, ha="left", fontsize=12.5)
fig.savefig(OUT/"B2c_tilt.png"); plt.close()

DZI.update(model=model, dz=DZrep, tilt_mm_per_m=float(coef[1]*1000),
           tilt_over_hedge=float(coef[1]*np.ptp(hy)),
           road_covers_hedge=float(cover), resid_iqr=float(rob(res_p if use_plane
                                                              else res_c)))
(OUT/"ennezat_dz.json").write_text(json.dumps(DZI, indent=2))
(OUT/"REPORT_B2c.md").write_text(
    "# Stage B2c — Constant or tilted offset\n\n```\n"+"\n".join(L)+"\n```\n",
    encoding="utf-8")
P()
P("written: ennezat_dz.json (updated), REPORT_B2c.md, B2c_tilt.png")
P("Next: python preparation/04_export_shared_clouds.py")