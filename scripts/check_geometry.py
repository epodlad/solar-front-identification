"""Projection illustration, recovered-model normals and STEREO visibility.

A selected model inclination is a constructed alignment. The ellipsoid fit
uses published model curves, not new multi-spacecraft tie points.
"""

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from geometry_core import observer, g1_xy, intersections, normal_record, point_on_ray

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["FRONT_OUTPUT"])
DEST = OUT / "geometry"
DEST.mkdir(exist_ok=True)
features = pd.read_csv(ROOT / "data/fan/inputs/display_feature_primary.csv")
waves = pd.read_csv(OUT / "fan/outputs/P100_waves.csv").set_index("family")
peak = g1_xy(features.peak_angle_deg.iloc[-1])
rear = g1_xy(features.rear_min_angle_deg.iloc[-1])
gap = float(np.linalg.norm(peak - rear))
age = 408.0
scale0 = 953.35 / 696000.0
fast = float(waves.loc["fast_plus", "speed_left_km_s"])
contact = float(waves.loc["entropy", "speed_left_km_s"])
cosine = gap / (scale0 * (fast - contact) * age)
if not 0 < cosine <= 1:
    raise RuntimeError("No real inclination satisfies the proposed alignment.")
scale = scale0 * cosine
summary = {
    "gap_arcsec": gap,
    "chosen_inclination_to_sky_deg": float(np.rad2deg(np.arccos(cosine))),
    "age_s": age,
    "contact_aligned_by_construction": True,
    "rarefaction_behind_contact_arcsec": sorted(
        [
            float(scale * (contact - waves.loc["fast_minus", k]) * age)
            for k in ["speed_left_km_s", "speed_right_km_s"]
        ]
    ),
}
(DEST / "projection_illustration.json").write_text(json.dumps(summary, indent=2) + "\n")


def load_observer(file):
    h = json.loads((ROOT / "data/geometry" / file).read_text())
    return observer(
        float(h["HGLN_OBS"]), float(h["HGLT_OBS"]), float(h["DSUN_OBS"]) / 1e6
    )


earth = load_observer("AIA_160042_header.json")
parameters = json.loads((DEST / "ELLIPSOID_FIT.json").read_text())["joint_parameters"]
records = [
    normal_record(parameters, theta, height, earth)
    for height in [0, 50, 100]
    for theta in intersections(parameters, height, earth)
]
(DEST / "local_normals.json").write_text(json.dumps(records, indent=2) + "\n")
earth = load_observer("AIA_reference_header.json")
stereo = load_observer("EUVI_header.json")
rows = []
for label in ["peak_angle_deg", "rear_min_angle_deg"]:
    for frame, r in features.iterrows():
        for height in [0, 50, 100, 200]:
            p = point_on_ray(*g1_xy(r[label]), height, earth) * 696.0
            v = stereo["xyz"] - p
            fraction = np.clip(-p @ v / (v @ v), 0, 1)
            clearance = float(np.linalg.norm(p + fraction * v) - 696.0)
            rows.append(
                {
                    "feature": label,
                    "frame": frame + 1,
                    "height_Mm": height,
                    "clearance_Mm": clearance,
                    "visible": clearance >= -1e-6,
                }
            )
pd.DataFrame(rows).to_csv(DEST / "stereo_visibility.csv", index=False)
print(
    json.dumps(
        {
            "projection": summary,
            "visible_points": sum(r["visible"] for r in rows),
            "tested_points": len(rows),
        },
        indent=2,
    )
)
