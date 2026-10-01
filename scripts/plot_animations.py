"""Render four observational movies. No solver, tracking, or temporal interpolation.

Run: python run.py animations (from the repository root)
Requires numpy, pandas, matplotlib, Pillow; ffmpeg for MP4 exports.
"""

from pathlib import Path
import json, shutil, os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import AsinhNorm, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
I = ROOT / "data/animations"
O = Path(os.environ.get("FRONT_OUTPUT", ROOT / "results")) / "movies"
O.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
)
C = {"fast": "#f2a329", "contact": "#de499c", "rare": "#9074cb", "other": "#39a296"}
META = {
    "Creator": f"Python/Matplotlib {matplotlib.__version__}",
    "Producer": None,
    "CreationDate": None,
    "ModDate": None,
    "Title": None,
    "Author": None,
}
INFO = {}


def base(title, subtitle, footer, figsize=(10.8, 6.5)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.83, bottom=0.18, wspace=0.23)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.97, color="#203c4d")
    fig.text(0.5, 0.9, subtitle, ha="center", fontsize=11, color="#455c68")
    fig.text(0.5, 0.033, footer, ha="center", fontsize=9, color="#435867")
    for ax in axs:
        ax.set_xlabel("Solar X [arcsec]")
        ax.set_ylabel("Solar Y [arcsec]")
    return fig, axs


def colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.15)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def save_movie(name, indices, times, render, seconds_per_epoch=0.6):
    """Repeated video frames hold each observation; no intermediate image is made."""
    images = []
    durations = []
    poster = len(indices) // 2
    first = render(indices[0])
    size = first.get_size_inches()
    plt.close(first)
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        dummy = plt.figure(figsize=size)
        writer = FFMpegWriter(
            fps=10,
            codec="libx264",
            bitrate=2400,
            metadata={"comment": f"Python/Matplotlib {matplotlib.__version__}"},
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-map_metadata",
                "-1",
            ],
        )
        writer.setup(dummy, str(O / (name + ".mp4")), dpi=120)
        plt.close(dummy)
    for k, j in enumerate(indices):
        fig = render(j)
        fig.canvas.draw()
        images.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).copy()
        )
        hold = (
            max(1.2, seconds_per_epoch)
            if k in [0, len(indices) - 1]
            else seconds_per_epoch
        )
        durations.append(round(hold * 1000))
        if k == poster:
            fig.savefig(
                O / (name + ".png"),
                dpi=120,
                metadata={"Software": f"Python/Matplotlib {matplotlib.__version__}"},
            )
            fig.savefig(O / (name + ".pdf"), metadata=META)
        if has_ffmpeg:
            writer.fig = fig
            for _ in range(round(hold * 10)):
                writer.grab_frame(facecolor="white")
        plt.close(fig)
    if has_ffmpeg:
        writer.finish()
    images[0].save(
        O / (name + ".gif"),
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    INFO[name] = {
        "observation_count": len(indices),
        "times_UTC": list(map(str, times)),
        "duration_ms": durations,
        "temporal_interpolation": False,
        "tracking_refitted": False,
        "model_refitted": False,
    }
    print(name, "done", flush=True)


e11 = json.loads((I / "e11_display.json").read_text())


def render_e11(j):
    f = e11["frames"][j]
    extent = e11["display"]["coordinates_arcsec"]
    fig, axes = base(
        "13 June 2010 | AIA 193 Å | E11",
        f["utc"][11:22] + " UT",
        "Retained brightness-crest fits | Colors mark image features, not identified MHD waves.\nNASA/SDO and the AIA science team",
    )
    for ax, key, title in zip(
        axes,
        ["original", "base_a"],
        ["Brightness", "Base difference | reference 05:35:08.07 UT"],
    ):
        ax.imshow(
            Image.open(I / f["images"][key]),
            origin="upper",
            extent=extent,
            interpolation="none",
        )
        for label, col in [("inner", "#21a2a7"), ("outer", C["fast"])]:
            if label not in f["overlays"]:
                continue
            xy = np.asarray(f["overlays"][label])
            x = extent[0] + xy[:, 0] / (e11["width"] - 1) * (extent[1] - extent[0])
            y = extent[3] - xy[:, 1] / (e11["height"] - 1) * (extent[3] - extent[2])
            ax.plot(x, y, color="white", lw=3)
            ax.plot(x, y, color=col, lw=1.6, label=label.capitalize() + " crest")
        ax.set(xlim=(940, 1190), ylim=(-650, -410), title=title)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", fontsize=9, framealpha=0.92)
    colorbar(
        fig,
        axes[1],
        plt.cm.ScalarMappable(norm=Normalize(-12, 12), cmap="RdBu_r"),
        "Brightness change [DN s$^{-1}$]",
    )
    # Equal image-panel heights, with room for the right colorbar.
    pos = axes[1].get_position()
    p = axes[0].get_position()
    axes[0].set_position([p.x0, pos.y0, p.width, pos.height])
    return fig


e05 = np.load(ROOT / "data/imaging/E05_registered_grid.npz")
e05boxes = json.loads((ROOT / "data/imaging/E05_apertures.json").read_text())["records"]
e05_t0 = np.datetime64("2011-02-16T14:28:44.840000")
e05_times = e05_t0 + np.rint(
    (e05["mid_seconds"] - e05["mid_seconds"][17]) * 1e6
).astype("timedelta64[us]")


def render_e05(j):
    rounded = np.datetime64(
        int(np.rint(e05_times[j].astype("int64") / 10000)) * 10000, "us"
    )
    t = np.datetime_as_string(rounded, unit="ms")
    fig, axes = base(
        "16 February 2011 | AIA 193 Å | E05",
        t[11:22] + " UT",
        "Yellow boxes: alternative EIS sampling positions | No unique front is assigned.\nNASA/SDO and the AIA science team",
    )
    ext = [
        e05["x_arcsec"][0],
        e05["x_arcsec"][-1],
        e05["y_arcsec"][0],
        e05["y_arcsec"][-1],
    ]
    im = axes[0].imshow(
        e05["DN_per_s"][j],
        origin="lower",
        extent=ext,
        cmap="magma",
        norm=AsinhNorm(linear_width=60, vmin=95, vmax=310),
        interpolation="none",
    )
    colorbar(fig, axes[0], im, "Brightness [DN s$^{-1}$]")
    im = axes[1].imshow(
        e05["DN_per_s"][j] - e05["DN_per_s"][1],
        origin="lower",
        extent=ext,
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
        interpolation="none",
    )
    colorbar(fig, axes[1], im, "Brightness change [DN s$^{-1}$]")
    for ax in axes:
        for b in e05boxes:
            ax.add_patch(
                Rectangle(
                    (b["centre_x_arcsec"] - 1, b["centre_y_arcsec"] - 2.5),
                    2,
                    5,
                    fill=False,
                    edgecolor="#f4d23c",
                    lw=0.8,
                )
            )
    axes[0].set_title("Brightness")
    axes[1].set_title("Base difference | reference 14:25:32.84 UT")
    return fig


su = np.load(I / "suvi.npz")
sp = pd.read_csv(ROOT / "reference_results/suvi/primary_positions.csv")
sg = json.loads((ROOT / "data/suvi/geometry_protocol.json").read_text())
anchor = np.array(sg["anchor_xy_arcsec"])
normal = np.array([np.cos(np.deg2rad(105)), np.sin(np.deg2rad(105))])
tangent = np.array([normal[1], -normal[0]])


def point(s):
    return anchor + np.asarray(s)[..., None] * normal


def interval(ax, lo, hi, fc, ec, alpha=1, hatch=None, ls="-"):
    a, b = point(lo), point(hi)
    width = 25 if fc == "none" else 48
    ax.add_patch(
        Polygon(
            [
                a - width * tangent,
                a + width * tangent,
                b + width * tangent,
                b - width * tangent,
            ],
            fc=fc,
            ec=ec,
            alpha=alpha,
            hatch=hatch,
            ls=ls,
            lw=1.5,
        )
    )


def render_suvi(j):
    row = sp.iloc[j]
    dt = su["times_UTC"][j + 1][11:22]
    fig, axes = base(
        "10 September 2017 | SUVI 195 Å",
        dt + " UT | P100 geometric test",
        "C and E set one common model scale; purple shows the required fast rarefaction.\nConditional association, not an observed complete fan | NOAA/NCEI, GOES-16 SUVI",
        figsize=(10.4, 8),
    )
    ext = [su["x_arcsec"][0], su["x_arcsec"][-1], su["y_arcsec"][0], su["y_arcsec"][-1]]
    a = su["radiance"][j + 1].astype(float)
    im = axes[0].imshow(
        np.log10(np.maximum(a, 1e-8)),
        origin="lower",
        extent=ext,
        cmap="magma",
        vmin=-2,
        vmax=0.8,
    )
    colorbar(fig, axes[0], im, "log$_{10}$ radiance [W m$^{-2}$ sr$^{-1}$]")
    im = axes[1].imshow(
        a - su["radiance"][0].astype(float),
        origin="lower",
        extent=ext,
        cmap="RdBu_r",
        norm=AsinhNorm(linear_width=0.025, vmin=-0.5, vmax=0.5),
    )
    cb = colorbar(fig, axes[1], im, "Radiance change [W m$^{-2}$ sr$^{-1}$]")
    cb.set_ticks([-0.5, -0.1, 0, 0.1, 0.5])
    cb.set_ticklabels(["−0.50", "−0.10", "0", "0.10", "0.50"])
    for ax in axes:
        xy = point(np.linspace(-790, 470, 200))
        ax.plot(*xy.T, c="#b1bbc1", lw=0.7)
        interval(ax, -400, -100, "none", "#e0bd1b", ls=":")
        interval(
            ax,
            row.required_head_arcsec,
            row.required_tail_arcsec,
            C["rare"],
            C["rare"],
            alpha=0.48,
            hatch="///",
        )
        for name, s, col, mark in [
            ("C", row.C_arcsec, C["fast"], "o"),
            ("E", row.E_arcsec, C["contact"], "s"),
        ]:
            xy = point(s)
            ax.scatter(*xy, s=40, c=col, marker=mark, ec="white", lw=0.6, zorder=5)
            ax.annotate(
                name,
                xy,
                xytext=(10, 0),
                textcoords="offset points",
                color=col,
                fontweight="bold",
                fontsize=12,
                bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1},
            )
        ax.text(
            *point(-250),
            " B",
            color="#725a00",
            fontweight="bold",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1},
        )
        ax.set(xlim=(825, 1470), ylim=(-610, 750))
    axes[0].set_title("Calibrated radiance")
    axes[1].set_title("Base difference | reference 15:30:24 UT")
    return fig


aia = np.load(ROOT / "data/fan/inputs/spherical_cut_cache.npz")
obs = pd.read_csv(ROOT / "data/fan/inputs/display_feature_primary.csv")
pred = pd.read_csv(ROOT / "reference_results/fan/predicted_positions.csv")
Rsun = 953.35
r0 = np.array([np.sqrt(1 - (-150 / Rsun) ** 2), -150 / Rsun, 0])
end = np.array([0, -750 / Rsun, np.sqrt(1 - (-750 / Rsun) ** 2)])
tt = end - (end @ r0) * r0
tt /= np.linalg.norm(tt)
qq = np.cross(r0, tt)


def geo(theta, offset=0):
    th = np.deg2rad(np.atleast_1d(theta))
    off = np.deg2rad(offset)
    return (
        Rsun
        * (
            np.cos(th)[:, None] * r0
            + np.sin(th)[:, None] * (tt * np.cos(off) + qq * np.sin(off))
        )[:, :2]
    )


def render_aia(j):
    idx = j + 2
    time = aia["times"][idx]
    fig, axes = base(
        "10 September 2017 | AIA 193 Å | G1",
        time[11:22] + " UT | fixed P100 prediction",
        "Circle: selected image maximum | Colored marks: conditional model positions.\nFast shock aligned in the first frame only | NASA/SDO and the AIA science team",
    )
    fig.subplots_adjust(top=0.78)
    ext = [
        aia["x_arcsec"][0],
        aia["x_arcsec"][-1],
        aia["y_arcsec"][0],
        aia["y_arcsec"][-1],
    ]
    im = axes[0].imshow(
        aia["images"][idx], origin="lower", extent=ext, cmap="magma", vmin=0, vmax=255
    )
    colorbar(fig, axes[0], im, "Displayed brightness")
    im = axes[1].imshow(
        aia["images"][idx].astype(float) - aia["images"][idx - 1].astype(float),
        origin="lower",
        extent=ext,
        cmap="RdBu_r",
        vmin=-18,
        vmax=18,
    )
    colorbar(fig, axes[1], im, "Change in displayed brightness")
    rows = pred[(pred.case == "P100") & (pred.UTC == time)]
    assert len(rows) == 7
    for ax in axes:
        g = geo(np.linspace(8, 80, 300))
        ax.plot(*g.T, c="#cad2d5", lw=0.7)
        for _, r in rows.iterrows():
            col = (
                C["contact"]
                if r.family == "entropy"
                else C["fast"] if r.family == "fast_plus" else C["other"]
            )
            if r.structure == "rarefaction":
                a = geo(np.linspace(r.theta_left_deg, r.theta_right_deg, 40), -1.5)
                b = geo(np.linspace(r.theta_left_deg, r.theta_right_deg, 40), 1.5)
                ax.fill(
                    np.r_[a[:, 0], b[::-1, 0]],
                    np.r_[a[:, 1], b[::-1, 1]],
                    fc=C["rare"],
                    alpha=0.65,
                )
            else:
                a, b = geo(r.theta_left_deg, -2.5)[0], geo(r.theta_left_deg, 2.5)[0]
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    c=col,
                    lw=2 if r.family in ["entropy", "fast_plus"] else 1,
                )
        a = geo(obs.peak_angle_deg.iloc[j])[0]
        ax.scatter(*a, facecolor="none", edgecolor="#00bad4", s=90, lw=1.8)
        ax.set(xlim=(250, 990), ylim=(-830, -190))
    axes[0].set_title("Brightness and spherical sampling path")
    axes[1].set_title(
        "Running difference | " + aia["times"][idx - 1][11:22] + " UT reference"
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=C[k], lw=3, label=l)
            for k, l in [
                ("fast", "Fast shock"),
                ("contact", "Contact"),
                ("rare", "Fast rarefaction"),
                ("other", "Other model waves"),
            ]
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.872),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    return fig


if __name__ == "__main__":
    save_movie(
        "Movie_S1_E11",
        list(range(len(e11["frames"]))),
        [f["utc"] for f in e11["frames"]],
        render_e11,
        0.5,
    )
    save_movie("Movie_S2_E05", list(range(1, 30)), e05_times[1:], render_e05, 0.35)
    save_movie("Movie_S3_SUVI", list(range(5)), su["times_UTC"][1:], render_suvi, 1.2)
    save_movie("Movie_S4_AIA_G1", list(range(5)), aia["times"][2:], render_aia, 1.2)
    (O / "ANIMATION_METADATA.json").write_text(json.dumps(INFO, indent=2))
