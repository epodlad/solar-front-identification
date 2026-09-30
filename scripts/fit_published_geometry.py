"""Recover an approximate ellipsoid from digitized Hu et al. (2019) model curves.

Adapted for standalone paths; the original published parameter values and a
new triangulated observation are not inferred by this fit.
"""

from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.optimize import least_squares
import os

P = Path(__file__).resolve().parents[1]
O = Path(os.environ["FRONT_OUTPUT"]) / "geometry"
O.mkdir(exist_ok=True)
z = np.load(P / "data/geometry/digitized_points.npz")
Bearth = 7.239
Lsta = -127.93585
Bsta = -4.24268


def basis(L, B):
    L, B = np.deg2rad([L, B])
    n = np.array([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)])
    w = np.array([-np.sin(L), np.cos(L), 0])
    north = np.cross(n, w)
    return np.array([w, north])


mat = {
    "a": basis(0, Bearth),
    "b": basis(Lsta, Bsta),
    "c": basis(0, Bearth),
    "d": basis(Lsta, Bsta),
}


def unit(L, B):
    L, B = np.deg2rad([L, B])
    return np.array([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)])


def circle_res(p, sets):
    L, B, mu = p
    e = unit(L, B)
    out = []
    for lab in ["a", "b"]:
        uv = sets[lab + "_cyan"]
        pr = mat[lab] @ e
        M = (1 - mu * mu) * (np.eye(2) - np.outer(pr, pr))
        inv = np.linalg.inv(M)
        delta = uv - mu * pr
        grad = delta @ inv
        out.extend(
            (
                (np.sum(delta * grad, axis=1) - 1)
                / (2 * np.linalg.norm(grad, axis=1))
                * sets[lab + "_circle"][2]
            ).tolist()
        )
    return np.array(out)


sets = {k: z[k] for k in z.files}
cf = least_squares(
    circle_res,
    [97, -9, 0.75],
    args=(sets,),
    bounds=([75, -30, 0.35], [120, 15, 0.95]),
    loss="soft_l1",
    f_scale=1,
)
phi = np.linspace(0, 2 * np.pi, 72, endpoint=False)
dirs = np.array([np.cos(phi), np.sin(phi)]).T
support = {
    lab: np.quantile(sets[lab + "_yellow"] @ dirs.T, 0.995, axis=0)
    for lab in ["c", "d"]
}


def unpack(p):
    L, B, mu, a, b = p
    rad = 1 - (1 - mu * mu) / (b * b)
    if rad <= 0:
        return None
    d = mu + a * np.sqrt(rad)
    e = unit(L, B)
    return d, a, b, e


def residual(p, sets, support):
    val = unpack(p)
    if val is None:
        return np.ones(len(circle_res(p[:3], sets)) + 144) * 1e4
    d, a, b, e = val
    out = list(circle_res(p[:3], sets))
    # Balance the high-resolution intersection geometry with one residual per support direction.
    out = np.array(out) / np.sqrt(len(out) / 144)
    yy = []
    for lab in ["c", "d"]:
        pr = mat[lab] @ e
        M = b * b * np.eye(2) + (a * a - b * b) * np.outer(pr, pr)
        pred = dirs @ (d * pr) + np.sqrt(np.sum((dirs @ M) * dirs, axis=1))
        yy.extend(((pred - support[lab]) * sets[lab + "_circle"][2]).tolist())
    return np.r_[out, yy]


fit = least_squares(
    residual,
    [*cf.x, 1.0, 1.2],
    args=(sets, support),
    bounds=([75, -30, 0.35, 0.3, 0.96], [120, 15, 0.95, 2.5, 2.5]),
    loss="soft_l1",
    f_scale=1.5,
    max_nfev=300,
)
d, a, b, e = unpack(fit.x)
result = {
    "parameter_order": [
        "axis_longitude_W_deg",
        "axis_latitude_deg",
        "surface_intersection_mu",
        "radial_semiaxis_Rsun",
        "lateral_semiaxis_Rsun",
    ],
    "cyan_only_parameters": cf.x.tolist(),
    "cyan_only_RMS_pixels": float(np.sqrt(np.mean(circle_res(cf.x, sets) ** 2))),
    "joint_parameters": fit.x.tolist(),
    "center_distance_Rsun": float(d),
    "axis_vector_HGS": e.tolist(),
    "success": bool(fit.success),
    "joint_residual_RMS_pixels_weighted": float(
        np.sqrt(np.mean(residual(fit.x, sets, support) ** 2))
    ),
    "joint_cyan_RMS_pixels": float(np.sqrt(np.mean(circle_res(fit.x[:3], sets) ** 2))),
    "source": "Approximate digitization of Hu et al.2019 Fig2a-d at ~16:00UT; not original author parameters.",
    "q_outer_support": 0.995,
}
(O / "ELLIPSOID_FIT.json").write_text(json.dumps(result, indent=2) + "\n")
np.savez_compressed(
    O / "fit_supports.npz",
    directions=dirs,
    c_support=support["c"],
    d_support=support["d"],
)
print(json.dumps(result, indent=2))
