"""Reproduce Figures 2--5 from calculated states and aligned measurements."""

from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

P = Path(__file__).resolve().parents[1]
D = Path(os.environ["FRONT_OUTPUT"])
F = D / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "axes.linewidth": 0.65,
    }
)
cols = {
    "fast": "#cb671f",
    "slow": "#3273a8",
    "alfven": "#288a80",
    "contact": "#b12b78",
    "rare": "#7760a7",
}
cc = {"P080": "#187fa0", "P100": "#6c4997", "P120": "#b46b24"}


def save(fig, n):
    fig.savefig(
        F / (n + ".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "Python/Matplotlib " + matplotlib.__version__,
            "Producer": None,
            "CreationDate": None,
            "ModDate": None,
            "Title": None,
            "Author": None,
        },
    )
    fig.savefig(
        F / (n + ".png"),
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Python/Matplotlib " + matplotlib.__version__},
    )
    plt.close(fig)


# 1: every complete pair is distinguished by the flow position among characteristics.
fig, axs = plt.subplots(3, 1, figsize=(7.1, 5.2), layout="constrained")
values = []
for case in json.loads((D / "single_fronts/checks.json").read_text()):
    values.append(
        (
            case["label"],
            [
                [q["slow_km_s"], q["alfven_n_km_s"], q["fast_km_s"], q["flow_km_s"]]
                for q in [case["upstream"], case["downstream"]]
            ],
        )
    )
for k, (ax, (title, v)) in enumerate(zip(axs, values)):
    for y, row in zip([1, 0], v):
        ax.hlines(y, 0, 760, color="#d8dde0", lw=0.8)
        for x, c, marker in zip(
            row,
            [cols["slow"], cols["alfven"], cols["fast"], "#20272d"],
            ["|", "|", "|", "D"],
        ):
            ax.plot(
                x, y, marker, color=c, ms=12 if marker == "|" else 5, mew=2, ls="none"
            )
    ax.set(
        xlim=(-5, 760),
        ylim=(-0.45, 1.5),
        yticks=[0, 1],
        yticklabels=["Downstream", "Upstream"],
        title=f"({chr(97+k)}) {title}",
    )
    ax.set_xticks([0, 150, 300, 450, 600, 750])
    ax.grid(axis="x", alpha=0.13)
    ax.set_xlabel("Normal flow and characteristic speeds [km s$^{-1}$]")
handles = [
    Line2D([], [], marker=m, ls="", color=c, ms=8, label=s)
    for m, c, s in [
        ("|", cols["slow"], "Slow"),
        ("|", cols["alfven"], "Normal Alfvén"),
        ("|", cols["fast"], "Fast"),
        ("D", "#20272d", "Flow magnitude"),
    ]
]
axs[0].legend(
    handles=handles,
    ncol=4,
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.51, 1.56),
    frameon=False,
)
save(fig, "figure01_local_pairs")
# 2: sampling + regression.
a = np.load(D / "solo/aligned_5min.npz")
df = pd.read_csv(D / "solo/primary_15s_samples.csv")
ch = json.loads((D / "solo/checks.json").read_text())
co = ch["comparisons"][1]
origin = np.datetime64("2021-08-30T10:21:00")
t = (a["time"] - origin) / np.timedelta64(1, "s")
tm = (a["raw_MAG_time"] - origin) / np.timedelta64(1, "s")
vht = np.array([co["V_HT_R_km_s"], co["V_HT_T_km_s"], co["V_HT_N_km_s"]])
mu = 4 * np.pi * 1e-7
mp = 1.67262192369e-27
B = df[[f"B_{s}_nT" for s in "RTN"]].values
V = df[[f"V_{s}_km_s" for s in "RTN"]].values
va = B * 1e-9 / np.sqrt(mu * mp * df.N_cm3.values[:, None] * 1e6) / 1000
w = V - vht
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.9), layout="constrained")
cvec = ["#be573f", "#22847f", "#596fb2"]
for j, (cl, lbl) in enumerate(zip(cvec, "RTN")):
    axs[0, 0].plot(tm, a["raw_MAG_B"][:, j], color=cl, lw=0.75, label=lbl)
    axs[0, 0].plot(t, a["B"][:, j], ".", color=cl, ms=4)
    axs[0, 1].errorbar(
        t,
        a["V"][:, j] - vht[j],
        xerr=0.5,
        fmt="o",
        ms=3,
        color=cl,
        elinewidth=1.2,
        lw=0.8,
    )
    axs[1, 0].scatter(
        va[:, j],
        w[:, j],
        c=cl,
        s=22,
        label=lbl,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
for ax in axs[0]:
    ax.axvspan(24, 28, color="#6d7380", alpha=0.14)
    ax.set(xlim=(7, 46), xlabel="Seconds after 10:21:00 UT")
    ax.grid(alpha=0.12)
axs[0, 0].set(ylabel="Magnetic field [nT]", title="(a) MAG and PAS acquisition times")
axs[0, 0].legend(ncol=3, frameon=False, fontsize=8)
axs[0, 1].set(
    ylabel="$u - V_{HT}$ [km s$^{-1}$]", title="(b) Plasma vectors in the HT frame"
)
xx = np.array([-45, 45])
axs[1, 0].plot(xx, xx, "--", c="#929aa1", lw=0.9)
axs[1, 0].plot(xx, co["slope"] * xx + co["intercept_km_s"], c="#283640", lw=1)
axs[1, 0].set(
    xlabel="$B / \\sqrt{\\mu_0 m_p n_p}$ [km s$^{-1}$]",
    ylabel="$u - V_{HT}$ [km s$^{-1}$]",
    title="(c) Primary interval: 15 s padding",
    xlim=(-44, 42),
    ylim=(-43, 40),
)
axs[1, 0].text(
    0.04,
    0.94,
    f"Slope {co['slope']:.2f}; pooled r = {co['pooled_r']:.2f}",
    transform=axs[1, 0].transAxes,
    va="top",
    fontsize=8,
)
axs[1, 0].grid(alpha=0.12)
c = ch["comparisons"]
axs[1, 1].plot(
    [x["padding_s"] for x in c], [x["slope"] for x in c], "o-", color="#663e83", lw=1.4
)
for x in c:
    axs[1, 1].annotate(
        f"n = {x['PAS_moments']}",
        (x["padding_s"], x["slope"]),
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
axs[1, 1].axhline(1, color="#8a949d", lw=0.9)
axs[1, 1].set(
    xlabel="Padding on each side of core [s]",
    ylabel="Pooled Walén slope",
    title="(d) Interval sensitivity",
    xticks=[10, 15, 20],
    xlim=(8, 22),
    ylim=(0.67, 1.07),
)
axs[1, 1].grid(alpha=0.12)
save(fig, "figure02_solar_orbiter")
# 3: exact calculated wave positions in distance-time space.
fig, axs = plt.subplots(3, 1, figsize=(7.1, 7.4), sharex=True, layout="constrained")
time = np.linspace(0, 410, 100)
labels = ["F−", "A−", "S−", "C", "S+", "A+", "F+"]
for ax, (cid, col) in zip(axs, cc.items()):
    wv = pd.read_csv(D / f"fan/outputs/{cid}_waves.csv")
    for i, r in wv.iterrows():
        family = r.family.split("_")[0]
        cl = cols.get(family, cols["contact"])
        l = r.speed_left_km_s * time / 1000
        rgt = r.speed_right_km_s * time / 1000
        if r.structure == "rarefaction":
            ax.fill_betweenx(time, l, rgt, color=cols["rare"], alpha=0.27)
            ax.plot(l, time, color=cols["rare"], lw=0.9)
            ax.plot(rgt, time, color=cols["rare"], lw=0.9)
            if family == "fast":
                for vv in np.linspace(r.speed_left_km_s, r.speed_right_km_s, 6)[1:-1]:
                    ax.plot(
                        vv * time / 1000, time, color=cols["rare"], lw=0.4, alpha=0.6
                    )
        else:
            ax.plot(
                l,
                time,
                color=cl,
                lw=1.35,
                ls=(
                    ":"
                    if r.structure == "rotation"
                    else "--" if r.structure == "contact" else "-"
                ),
            )
        xpos = (r.speed_left_km_s + r.speed_right_km_s) / 2 * 0.41
        ax.text(
            xpos,
            421,
            labels[i] + (" (r)" if r.structure == "rarefaction" else ""),
            ha="center",
            va="bottom",
            fontsize=8,
            color=cols["rare"] if r.structure == "rarefaction" else cl,
        )
    ax.set_title(
        cid
        + "  "
        + {
            "P080": "LEFT pressure × 0.8",
            "P100": "reference states",
            "P120": "LEFT pressure × 1.2",
        }[cid],
        loc="left",
        fontsize=9,
        pad=9,
    )
    ax.set(
        ylim=(0, 470), yticks=[0, 200, 400], xlim=(-210, 340), ylabel="Model time [s]"
    )
    ax.grid(alpha=0.1)
axs[-1].set_xlabel("Distance from the initial interface [Mm]")
fig.legend(
    handles=[
        Line2D([], [], c="#4b5055", label="Shock"),
        Line2D([], [], c=cols["alfven"], ls=":", label="Alfvénic rotation"),
        Line2D([], [], c=cols["contact"], ls="--", label="Contact"),
        Patch(facecolor=cols["rare"], alpha=0.3, label="Rarefaction"),
    ],
    ncol=4,
    loc="outside lower center",
    frameon=False,
    fontsize=8,
)
save(fig, "figure03_three_fans")


# 4: model profiles from states and continuous rarefaction samples.
def profile(cid, xi):
    data = json.loads((D / f"fan/outputs/{cid}_solution.json").read_text())
    st = pd.read_csv(D / f"fan/outputs/{cid}_states.csv")
    wv = pd.read_csv(D / f"fan/outputs/{cid}_waves.csv")
    norm = data["normalization"]
    vel = norm["velocity_m_s"] / 1000
    ne = np.full_like(xi, st.ne_cm3.iloc[0])
    pp = np.full_like(xi, st.p_Pa.iloc[0])
    for k, r in wv.iterrows():
        m = xi >= r.speed_right_km_s
        ne[m] = st.ne_cm3.iloc[k + 1]
        pp[m] = st.p_Pa.iloc[k + 1]
        if r.structure == "rarefaction":
            samples = [
                s
                for s in data["rarefaction_samples_normalized"]
                if s["family"] == r.family
            ]
            sx = []
            for s in samples:
                a2 = 5 * s["p"] / (3 * s["rho"])
                B = np.array(s["B"])
                b2 = B @ B / s["rho"]
                bn = B[0] ** 2 / s["rho"]
                cf2 = (a2 + b2 + np.sqrt((a2 + b2) ** 2 - 4 * a2 * bn)) / 2
                c = np.sqrt(cf2 if r.family.startswith("fast") else a2 * bn / cf2)
                sx.append(
                    (s["u"][0] + (-1 if r.family.endswith("minus") else 1) * c) * vel
                )
            order = np.argsort(sx)
            sx = np.array(sx)[order]
            m = (xi >= r.speed_left_km_s) & (xi < r.speed_right_km_s)
            ns = np.array(
                [s["rho"] * norm["density_kg_m3"] / mp / 1e6 for s in samples]
            )[order]
            ps = np.array([s["p"] * norm["pressure_Pa"] for s in samples])[order]
            ne[m] = np.interp(xi[m], sx, ns)
            pp[m] = np.interp(xi[m], sx, ps)
    return ne / 1e8, pp / (2 * ne * 1e6 * 1.380649e-23) / 1e6


fig, axs = plt.subplots(
    2, 2, figsize=(7.2, 4.7), width_ratios=[2, 1], sharex="col", layout="constrained"
)
for cid, col in cc.items():
    for j, xi in enumerate(
        [np.linspace(-470, 805, 11000), np.linspace(338, 358, 5000)]
    ):
        ne, T = profile(cid, xi)
        axs[0, j].plot(xi, ne, c=col, lw=1.15, label=cid)
        axs[1, j].plot(xi, T, c=col, lw=1.15)
for ax in axs.flat:
    ax.grid(alpha=0.13)
axs[0, 0].set(ylabel="Density [$10^8$ cm$^{-3}$]", title="(a) Complete fan")
axs[1, 0].set(ylabel="Temperature [MK]", xlabel="$x/t$ [km s$^{-1}$]")
axs[0, 1].set(title="(b) Inner slow+ wave")
axs[1, 1].set(xlabel="$x/t$ [km s$^{-1}$]")
axs[0, 0].legend(ncol=3, frameon=False, fontsize=8)
save(fig, "figure04_model_states")
