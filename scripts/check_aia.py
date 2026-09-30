"""Repeat exploratory feature selection on the retained AIA ray samples.

The samples are display levels. They are used for positions and shapes,
not for a density, temperature or radiance inversion. G1 is the assumed
great-circle path. The search intervals were selected after image inspection.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["FRONT_OUTPUT"]) / "aia"
OUT.mkdir(exist_ok=True)
z = np.load(ROOT / "data/fan/inputs/ray_samples.npz")
cache = np.load(ROOT / "data/fan/inputs/spherical_cut_cache.npz")
angle, rays = z["angle_deg"], z["display_levels"]
rows = []
for method in ["median", "mean"]:
    for halfwidth in [0.75, 1.125, 1.5]:
        selected = abs(z["offset_deg"]) <= halfwidth + 1e-8
        profile = getattr(np, method)(rays[:, selected, :], axis=1)
        for sigma in [0.3, 0.6, 0.9]:
            smoothed = gaussian_filter1d(profile, sigma / 0.15, axis=1)
            for k in range(2, 7):
                difference = smoothed[k] - smoothed[k - 1]
                search = np.flatnonzero((angle >= 35) & (angle <= 65))
                peak = search[np.argmax(difference[search])]
                search = np.flatnonzero((angle >= 20) & (angle < angle[peak]))
                rear = search[np.argmin(difference[search])]
                rows.append(
                    dict(
                        method=method,
                        halfwidth_deg=halfwidth,
                        sigma_deg=sigma,
                        UTC=cache["times"][k],
                        peak_angle_deg=float(angle[peak]),
                        peak_delta_display_levels=float(difference[peak]),
                        rear_min_angle_deg=float(angle[rear]),
                        rear_min_delta_display_levels=float(difference[rear]),
                        rear_min_change_from_155105_display_levels=float(
                            smoothed[k, rear] - smoothed[0, rear]
                        ),
                    )
                )
table = pd.DataFrame(rows)
primary = table[
    (table.method == "median") & (table.halfwidth_deg == 1.5) & (table.sigma_deg == 0.6)
].copy()
retained = pd.read_csv(ROOT / "data/fan/inputs/display_feature_primary.csv")
for column in [
    "peak_angle_deg",
    "rear_min_angle_deg",
    "peak_delta_display_levels",
    "rear_min_delta_display_levels",
    "rear_min_change_from_155105_display_levels",
]:
    if not np.allclose(primary[column], retained[column], rtol=0, atol=1e-10):
        raise RuntimeError(f"Primary profile does not reproduce {column}.")
variants = []
for keys, g in table.groupby(["method", "halfwidth_deg", "sigma_deg"]):
    angles = g.peak_angle_deg.to_numpy()
    variants.append(
        dict(
            method=keys[0],
            halfwidth_deg=keys[1],
            sigma_deg=keys[2],
            net_shift_deg=float(angles[-1] - angles[0]),
            strictly_increasing=bool((np.diff(angles) > 0).all()),
        )
    )
span = (
    pd.to_datetime(primary.UTC.iloc[-1]) - pd.to_datetime(primary.UTC.iloc[0])
).total_seconds()
shift = float(primary.peak_angle_deg.iloc[-1] - primary.peak_angle_deg.iloc[0])
summary = dict(
    variant_count=len(variants),
    all_net_outward=all(v["net_shift_deg"] > 0 for v in variants),
    primary_shift_deg=shift,
    conditional_surface_speed_km_s=float(696000 * np.deg2rad(shift) / span),
    primary_rear_minima_above_earlier_reference=bool(
        (primary.rear_min_change_from_155105_display_levels > 0).all()
    ),
    variants=variants,
)
table.to_csv(OUT / "sensitivity.csv", index=False)
primary.to_csv(OUT / "primary.csv", index=False)
(OUT / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
