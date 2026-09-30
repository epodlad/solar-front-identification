"""Test the necessary common-geometry relation using measured feature tracks.

This script starts from the retained C/E measurements for twelve processing
choices. It does not redetect the features or assign an observed wave type.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["FRONT_OUTPUT"])
DEST = OUT / "suvi"
DEST.mkdir(exist_ok=True)
tracks = pd.read_csv(ROOT / "data/suvi/feature_positions.csv")
waves = pd.read_csv(OUT / "fan/outputs/P100_waves.csv").set_index("family")
contact = waves.loc["entropy", "speed_left_km_s"]
fast = waves.loc["fast_plus", "speed_left_km_s"]
for edge, column in [("head", "speed_left_km_s"), ("tail", "speed_right_km_s")]:
    ratio = (waves.loc["fast_minus", column] - contact) / (fast - contact)
    tracks[f"required_{edge}_arcsec"] = tracks.E_arcsec + ratio * (
        tracks.C_arcsec - tracks.E_arcsec
    )
tracks["aperture_overlap_arcsec"] = np.maximum(
    0.0,
    np.minimum(tracks.required_tail_arcsec, -100.0)
    - np.maximum(tracks.required_head_arcsec, -400.0),
)
tracks["inside_aperture_any"] = tracks.aperture_overlap_arcsec > 0
primary = tracks[
    (tracks.baseline == "15:30:24")
    & (tracks.sigma_arcsec == 5)
    & (tracks.ray_group == "five")
]
if len(primary) != 5 or len(tracks) != 60:
    raise RuntimeError("Expected five epochs and twelve processing variants.")
tracks.to_csv(DEST / "all_12_variants.csv", index=False)
primary.to_csv(DEST / "primary_positions.csv", index=False)
summary = {
    "last_epoch_outside_B_all_variants": bool(
        (tracks[tracks.UTC == primary.UTC.iloc[-1]].aperture_overlap_arcsec == 0).all()
    ),
    "primary_outside_B_after_first": bool(
        (primary.iloc[1:].aperture_overlap_arcsec == 0).all()
    ),
    "last_primary_rarefaction_arcsec": primary.iloc[-1][
        ["required_head_arcsec", "required_tail_arcsec"]
    ].to_list(),
    "scope": "Necessary geometric test of C=fast shock, E=contact and B=fast rarefaction; no new wave identification.",
}
(DEST / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
