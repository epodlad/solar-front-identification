"""Scientific display revision. No state, trajectory, or wave solution is fitted."""

from pathlib import Path
import json, warnings, hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

import os

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["FRONT_OUTPUT"])
D = ROOT / "data"
WORK = OUT / "figures"
WORK.mkdir(parents=True, exist_ok=True)
P = OUT
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
COL = {
    "fast": "#ee931e",
    "contact": "#db3c96",
    "rare": "#8e70cf",
    "alfven": "#288a80",
    "slow": "#3273a8",
}
META = {
    "Creator": "Python/Matplotlib " + matplotlib.__version__,
    "Producer": None,
    "CreationDate": None,
    "ModDate": None,
    "Author": None,
    "Title": None,
    "Subject": None,
    "Keywords": None,
}


def save(fig, name):
    temp_pdf = WORK / f"{name}_export.pdf"
    fig.savefig(temp_pdf, bbox_inches="tight", metadata=META)
    assert temp_pdf.stat().st_size > 0
    temp_pdf.replace(P / "figures" / f"{name}.pdf")
    fig.savefig(
        WORK / f"{name}.png",
        dpi=175,
        bbox_inches="tight",
        metadata={"Software": "Python/Matplotlib " + matplotlib.__version__},
    )
    plt.close(fig)


def label(ax, text):
    ax.set_title(text, loc="left", pad=7)


def tag(ax, text, x=0.025, y=0.975):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "#10212d", "edgecolor": "none", "alpha": 0.78, "pad": 3},
    )


def arrow(ax, text, point, xytext, color="white", size=8):
    ax.annotate(
        text,
        point,
        xytext=xytext,
        color=color,
        fontsize=size,
        bbox={"facecolor": "#10212d", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        zorder=10,
    )


def bar(fig, ax, im, text, ticks=None):
    cb = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.19,
        fraction=0.05,
        aspect=30,
        ticks=ticks,
    )
    cb.ax.tick_params(labelsize=7, pad=1)
    cb.set_label(text, fontsize=7, labelpad=2)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:g}" for v in ticks])


def axes_solar(ax):
    ax.set_xlabel("Solar X [arcsec]")
    ax.set_ylabel("Solar Y [arcsec]")


# Frozen E11 display, including the existing spatial smoothing and crest fits.
e11 = json.loads((D / "imaging/E11_display.json").read_text())
e11frame = next(x for x in e11["frames"] if x["index"] == 8)
e11extent = e11["display"]["coordinates_arcsec"]


def panel_e11(ax, difference=True, compact=False):
    key = "base_a" if difference else "original"
    raster = Image.open(D / "imaging" / e11frame["images"][key])
    ax.imshow(raster, origin="upper", extent=e11extent, interpolation="none")
    x0, x1, y0, y1 = e11extent
    for name, color in [("inner", "#158e99"), ("outer", COL["fast"])]:
        pix = np.array(e11frame["overlays"][name])
        xx = x0 + pix[:, 0] / (e11["width"] - 1) * (x1 - x0)
        yy = y1 - pix[:, 1] / (e11["height"] - 1) * (y1 - y0)
        ax.plot(xx, yy, c="white", lw=3)
        ax.plot(xx, yy, c=color, lw=1.4)
    ax.set(xlim=(940, 1190), ylim=(-650, -410))
    axes_solar(ax)
    tag(ax, "13 June 2010\n05:39:20.05 UT")
    arrow(ax, "Outer crest", (1118, -536), (1050, -620))
    arrow(ax, "Inner crest", (1033, -509), (950, -481))
    return (
        plt.cm.ScalarMappable(norm=Normalize(-12, 12), cmap="RdBu_r")
        if difference
        else None
    )


# Frozen E05 common-grid, exposure-normalized AIA data. Only subtraction is new.
e05path = D / "imaging/E05_registered_grid.npz"
e05 = np.load(e05path)
e05diff = e05["DN_per_s"][17] - e05["DN_per_s"][1]
info = json.loads((D / "imaging/E05_apertures.json").read_text())


def panel_e05(ax, difference=True):
    extent = [
        e05["x_arcsec"][0],
        e05["x_arcsec"][-1],
        e05["y_arcsec"][0],
        e05["y_arcsec"][-1],
    ]
    if difference:
        im = ax.imshow(
            e05diff,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-30,
            vmax=30,
            interpolation="none",
        )
    else:
        im = ax.imshow(
            e05["DN_per_s"][17],
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=AsinhNorm(linear_width=60, vmin=95, vmax=310),
            interpolation="none",
        )
    for r in info["records"]:
        ax.add_patch(
            Rectangle(
                (r["centre_x_arcsec"] - 1, r["centre_y_arcsec"] - 2.5),
                2,
                5,
                fill=False,
                edgecolor="#ffd641",
                lw=0.7,
            )
        )
    axes_solar(ax)
    tag(ax, "16 February 2011\n14:28:44.84 UT")
    arrow(ax, "EIS sampling region", (449, 32), (394, -16))
    return im


# SUVI is resampled using each exposure's WCS onto the same declared grid.
z = pd.read_csv(OUT / "suvi/primary_positions.csv")
geom = json.loads((D / "suvi/geometry_protocol.json").read_text())
anchor = np.array(geom["anchor_xy_arcsec"])
n = np.array([np.cos(np.deg2rad(105)), np.sin(np.deg2rad(105))])
tangent = np.array([n[1], -n[0]])


def cutxy(s):
    return anchor + np.asarray(s)[..., None] * n


sx = np.arange(670, 1471, 2.5)
sy = np.arange(-610, 751, 2.5)
X, Y = np.meshgrid(sx, sy)


def read_suvi(stamp):
    f = next((D / "suvi").glob(f"*_{stamp}_*.fits.gz"))
    with fits.open(f) as hd:
        im = hd[0].data.astype(float)
        h = hd[0].header.copy()
    for k in ["OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z"]:
        if isinstance(h.get(k), str):
            h.pop(k)
    h.pop("CROTA", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = WCS(h, naxis=2)
    px, py = w.all_world2pix(X / 3600, Y / 3600, 0)
    return map_coordinates(im, [py, px], order=1, mode="constant", cval=np.nan)


suvi = read_suvi("155824")
suvi_base = read_suvi("153024")
suvi_diff = suvi - suvi_base
print("SUVI difference percentiles", np.nanpercentile(suvi_diff, [1, 5, 50, 95, 99]))


def panel_suvi(ax, difference=False, model=False, overview=False):
    extent = [sx[0], sx[-1], sy[0], sy[-1]]
    if difference:
        im = ax.imshow(
            suvi_diff,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            norm=AsinhNorm(linear_width=0.025, vmin=-0.5, vmax=0.5),
        )
    else:
        im = ax.imshow(
            np.log10(np.maximum(suvi, 1e-8)),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=-2,
            vmax=0.8,
        )
    line = cutxy(np.linspace(-790, 470, 300))
    ax.plot(*line.T, c="white" if not difference else "#34495a", lw=0.65, alpha=0.9)
    row = z.iloc[4]
    a, b = cutxy(-400), cutxy(-100)
    ax.add_patch(
        Polygon(
            [a - 25 * tangent, a + 25 * tangent, b + 25 * tangent, b - 25 * tangent],
            fill=False,
            ec="#f3d252",
            ls=":",
            lw=1.3,
        )
    )
    if model:
        a, b = cutxy(row.required_head_arcsec), cutxy(row.required_tail_arcsec)
        ax.add_patch(
            Polygon(
                [
                    a - 55 * tangent,
                    a + 55 * tangent,
                    b + 55 * tangent,
                    b - 55 * tangent,
                ],
                fc=COL["rare"],
                ec=COL["rare"],
                alpha=0.45,
                hatch="///",
            )
        )
    for name, s, color, marker in [
        ("C", row.C_arcsec, COL["fast"], "o"),
        ("E", row.E_arcsec, COL["contact"], "s"),
    ]:
        pt = cutxy(s)
        ax.scatter(*pt, s=30, c=color, marker=marker, ec="white", lw=0.65, zorder=8)
    arrow(
        ax,
        "C: outer feature" if overview else "C: fast-shock test",
        cutxy(row.C_arcsec),
        (730, 585),
        color="#ffe2a3",
    )
    arrow(
        ax,
        "E: inner edge" if overview else "E: contact test",
        cutxy(row.E_arcsec),
        (740, 310),
        color="#ffc1e5",
    )
    arrow(ax, "B: trailing aperture", cutxy(-250), (735, -90), color="#fff2a6")
    if model:
        arrow(
            ax, "Required\nfast rarefaction", cutxy(-695), (760, -435), color="#dfccff"
        )
    ax.set(xlim=(670, 1470), ylim=(-610, 750))
    axes_solar(ax)
    return im


# AIA display-level images. Floating-point subtraction prevents integer wrap.
aia = np.load(D / "fan/inputs/spherical_cut_cache.npz")
obs = pd.read_csv(D / "fan/inputs/display_feature_primary.csv")
pred = pd.read_csv(OUT / "fan/predicted_positions.csv")
comp = pd.read_csv(OUT / "fan/kinematic_comparison.csv")
R = 953.35
r0 = np.array([np.sqrt(1 - (-150 / R) ** 2), -150 / R, 0.0])
end = np.array([0.0, -750 / R, np.sqrt(1 - (-750 / R) ** 2)])
tt = end - (end @ r0) * r0
tt /= np.linalg.norm(tt)
qq = np.cross(r0, tt)


def geo(th, off=0):
    th = np.deg2rad(np.atleast_1d(th))
    o = np.deg2rad(off)
    return (
        R
        * (
            np.cos(th)[:, None] * r0
            + np.sin(th)[:, None] * (tt * np.cos(o) + qq * np.sin(o))
        )[:, :2]
    )


def panel_aia(ax, difference=False, model=False):
    extent = [
        aia["x_arcsec"][0],
        aia["x_arcsec"][-1],
        aia["y_arcsec"][0],
        aia["y_arcsec"][-1],
    ]
    if difference:
        delta = aia["images"][4].astype(float) - aia["images"][3].astype(float)
        im = ax.imshow(
            delta, origin="lower", extent=extent, cmap="RdBu_r", vmin=-18, vmax=18
        )
    else:
        im = ax.imshow(
            aia["images"][4],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=255,
        )
    ray = geo(np.linspace(8, 80, 300))
    ax.plot(*ray.T, c="#354a54" if difference else "white", lw=0.7)
    if model:
        rows = pred[(pred.case == "P100") & (pred.UTC == obs.UTC.iloc[2])]
        for _, r in rows.iterrows():
            cl = COL.get(r.family.split("_")[0], COL["contact"])
            if r.structure == "rarefaction":
                p1 = geo(np.linspace(r.theta_left_deg, r.theta_right_deg, 40), -1.5)
                p2 = geo(np.linspace(r.theta_left_deg, r.theta_right_deg, 40), 1.5)
                ax.fill(
                    np.r_[p1[:, 0], p2[::-1, 0]],
                    np.r_[p1[:, 1], p2[::-1, 1]],
                    c=COL["rare"],
                    alpha=0.7,
                )
            else:
                p1 = geo(r.theta_left_deg, -2.5)[0]
                p2 = geo(r.theta_left_deg, 2.5)[0]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    c=cl,
                    lw=1.8 if r.family in ["entropy", "fast_plus"] else 0.8,
                )
    pt = geo(obs.peak_angle_deg.iloc[2])[0]
    ax.scatter(*pt, facecolor="none", edgecolor="#00bcd4", s=42, lw=1.3, zorder=9)
    ax.set(xlim=(250, 990), ylim=(-830, -190))
    axes_solar(ax)
    return im


def panel_solo(ax):
    a = np.load(OUT / "solo/aligned_5min.npz")
    o = np.datetime64("2021-08-30T10:21:00")
    tm = (a["raw_MAG_time"] - o) / np.timedelta64(1, "s")
    for j, (color, name) in enumerate(zip(["#be573f", "#22847f", "#596fb2"], "RTN")):
        ax.plot(tm, a["raw_MAG_B"][:, j], c=color, lw=0.8, label=name)
    ax.axvspan(24, 28, color="#727b84", alpha=0.17)
    ax.set(xlim=(7, 46), xlabel="Seconds after 10:21:00 UT", ylabel="MAG [nT]")
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    ax.grid(alpha=0.12)


# Opening figure: each event in the paper, using the measured images/profiles.
fig = plt.figure(figsize=(7.3, 7.5), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.42])
ax = fig.add_subplot(gs[0, 0])
im = panel_e11(ax)
label(ax, "(a) E11 | AIA 193 Å base difference")
bar(fig, ax, im, "Change [DN s$^{-1}$]", [-12, 0, 12])
ax = fig.add_subplot(gs[0, 1])
im = panel_e05(ax)
label(ax, "(b) E05 | AIA 193 Å base difference")
bar(fig, ax, im, "Change [DN s$^{-1}$]", [-30, 0, 30])
ax = fig.add_subplot(gs[1, 0])
im = panel_suvi(ax, True, overview=True)
label(ax, "(c) SUVI 195 Å | 10 September 2017")
ax.set(ylim=(-260, 640), xlim=(670, 1470))
bar(fig, ax, im, "Radiance change [W m$^{-2}$ sr$^{-1}$]", [-0.5, 0, 0.5])
ax = fig.add_subplot(gs[1, 1])
im = panel_aia(ax, True)
label(ax, "(d) AIA 193 Å | 10 September 2017")
arrow(ax, "Selected maximum", geo(obs.peak_angle_deg.iloc[2])[0], (270, -410))
bar(fig, ax, im, "Change in display level", [-18, 0, 18])
ax = fig.add_subplot(gs[2, :])
panel_solo(ax)
label(ax, "(e) Solar Orbiter | 30 August 2021 | boundary core shaded")
save(fig, "figure00_observational_overview")

# SUVI: original radiance and a signed difference of the same exposure.
fig = plt.figure(figsize=(7.2, 6.6), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
ax = fig.add_subplot(gs[0, 0])
im = panel_suvi(ax, False, True)
label(ax, "(a) SUVI 195 Å | 15:58:24.76 UT")
bar(fig, ax, im, "log$_{10}$ radiance [W m$^{-2}$ sr$^{-1}$]", [-2, -1, 0])
ax = fig.add_subplot(gs[0, 1])
im = panel_suvi(ax, True, True)
label(ax, "(b) Same frame minus 15:30:24 UT")
bar(fig, ax, im, "Radiance change [W m$^{-2}$ sr$^{-1}$]", [-0.5, 0, 0.5])
ax = fig.add_subplot(gs[1, :])
ax.axhspan(-400, -100, color="#b3bbc1", alpha=0.23, label="Measured aperture B")
ax.fill_between(
    z.seconds,
    z.required_head_arcsec,
    z.required_tail_arcsec,
    color=COL["rare"],
    alpha=0.3,
    label="Required fast rarefaction",
)
ax.plot(
    z.seconds, z.C_arcsec, "o-", c=COL["fast"], label="C assigned to fast shock", ms=4
)
ax.plot(
    z.seconds, z.E_arcsec, "s--", c=COL["contact"], label="E assigned to contact", ms=4
)
ax.plot(z.seconds, z.required_head_arcsec, c=COL["rare"], lw=0.8)
ax.plot(z.seconds, z.required_tail_arcsec, c=COL["rare"], lw=0.8)
ax.set(
    xlabel="Time after 15:54:24.75 UT [s]",
    ylabel="Position along cut [arcsec]",
    ylim=(-815, 460),
    xlim=(-4, 245),
)
label(ax, "(c) One model geometry links all wave positions")
fig.legend(
    *ax.get_legend_handles_labels(),
    frameon=False,
    fontsize=7,
    ncol=2,
    loc="outside lower center",
)
ax.text(180, -230, "B", c="#53616b")
ax.grid(alpha=0.13)
save(fig, "figure05_suvi_constraint")

# AIA: show both the direct display and the signed preceding-frame change.
fig = plt.figure(figsize=(7.3, 5.3), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, 0])
panel_aia(ax, False, True)
label(ax, "(a) AIA 193 Å | 16:03:05.85 UT")
ax = fig.add_subplot(gs[0, 1])
im = panel_aia(ax, True, True)
label(ax, "(b) Minus 16:01:53.84 UT")
bar(fig, ax, im, "Display-level change", [-18, 0, 18])
pr = fig.add_subplot(gs[0, 2])
label(pr, "(c) Changes along G1")
for i, col in zip([0, 2, 4], ["#267f94", "#7760a7", "#b87527"]):
    k = i + 2
    pr.plot(
        aia["angle_deg"],
        aia["smoothed_levels"][k] - aia["smoothed_levels"][k - 1],
        c=col,
        lw=1,
        label=obs.UTC.iloc[i][11:19],
    )
    pr.plot(
        obs.peak_angle_deg.iloc[i],
        obs.peak_delta_display_levels.iloc[i],
        "o",
        c=col,
        ms=3,
    )
pr.axhline(0, c="#919aa2", lw=0.6)
pr.set(
    xlim=(15, 75),
    ylim=(-7, 10),
    xlabel="Angle along G1 [deg]",
    ylabel="Display-level change",
)
pr.legend(frameon=False, fontsize=6.5)
pr.grid(alpha=0.12)
ax = fig.add_subplot(gs[1, :])
x = np.array([0, 72, 144.01, 216, 288])
ax.plot(x, obs.peak_angle_deg, "o-", c="#157d96", ms=4, label="Selected image maximum")
ax.plot(
    x,
    obs.rear_min_angle_deg,
    "s:",
    c="#6a777f",
    ms=4,
    label="Selected trailing minimum",
)
for cid, col in {"P080": "#187fa0", "P100": "#6c4997", "P120": "#b46b24"}.items():
    q = comp[comp.case == cid]
    ax.plot(
        q.elapsed_s,
        q.predicted_conditional_fast_deg,
        "--",
        c=col,
        lw=1,
        label=cid + " fast shock",
    )
pp = pred[pred.case == "P100"]
cr = pp[pp.family == "entropy"]
rr = pp[pp.family == "fast_minus"]
ax.plot(x, cr.theta_left_deg, "--", c=COL["contact"], lw=1.3, label="P100 contact")
ax.fill_between(
    x,
    rr.theta_left_deg,
    rr.theta_right_deg,
    color=COL["rare"],
    alpha=0.2,
    label="P100 fast rarefaction",
)
ax.set(
    xlim=(-3, 293),
    ylim=(0, 66),
    xlabel="Time after 16:00:41.84 UT [s]",
    ylabel="Angle along G1 [deg]",
)
label(ax, "(d) First maximum aligned once; later model positions predicted")
ax.grid(alpha=0.12)
ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=3)
save(fig, "figure06_aia_comparison")


print("Created observational overview, SUVI and AIA figures.")
