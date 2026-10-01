"""Write a short human-readable guide to a completed reproduction run."""

import json
import os
from pathlib import Path

OUT = Path(os.environ["FRONT_OUTPUT"])


def read(path):
    return json.loads((OUT / path).read_text())


fronts = read("single_fronts/checks.json")
fans = read("fan/independent_model_checks.json")
solo = read("solo/checks.json")
suvi = read("suvi/checks.json")
aia = read("aia/checks.json")
lines = [
    "# Reproduction results",
    "",
    "This report was generated from the current run. Full-precision values are in the linked JSON and CSV files.",
    "",
    "## Single-front tests",
    "",
    "| Pair | Retained type | Maximum normalized flux difference | All checks |",
    "| --- | --- | --- | --- |",
]
for r in fronts:
    lines.append(
        f"| {r['case']} | {r['family']} shock | {r['max_normalized_flux_residual']:.2e} | {'Pass' if all(r['checks'].values()) else 'Fail'} |"
    )
lines += [
    "",
    "[Complete single-front results](single_fronts/checks.json). These supplied pairs are conditional models.",
    "",
    "## Connected fans",
    "",
    "| Model | Outer fast-shock speed (km/s) | Independent checks |",
    "| --- | --- | --- |",
]
for r in fans:
    lines.append(
        f"| {r['case']} | {r['fast_shock_speed_km_s']:.2f} | {'Pass' if r['independent_checks_pass'] else 'Fail'} |"
    )
lines += [
    "",
    "[Fan checks](fan/independent_model_checks.json). The inner slow+ connection is a rarefaction in P080 and a shock in P100/P120. Only the retained regular branch is tested.",
    "",
    "## Solar Orbiter",
    "",
    "| Padding per side (s) | PAS vectors | Walén slope | Pooled correlation |",
    "| --- | --- | --- | --- |",
]
for r in solo["comparisons"]:
    lines.append(
        f"| {r['padding_s']} | {r['PAS_moments']} | {r['slope']:.2f} | {r['pooled_r']:.2f} |"
    )
lines += [
    "",
    "[Solar Orbiter results](solo/checks.json). The relation is Alfvénic; one proton acquisition in the core does not provide a complete discontinuity classification.",
    "",
    "## Image comparisons",
    "",
    f"- SUVI: rarefaction outside aperture B in every final-epoch variant: **{suvi['last_epoch_outside_B_all_variants']}**.",
    f"- AIA: selected maximum shifts by **{aia['primary_shift_deg']:.2f}°**; assumed surface-feature speed **{aia['conditional_surface_speed_km_s']:.2f} km/s**.",
    f"- AIA: net outward motion in all **{aia['variant_count']}** processing combinations: **{aia['all_net_outward']}**.",
    "- No contact or rarefaction has been independently identified in these image tracks.",
    "",
    "[SUVI results](suvi/checks.json) · [AIA results](aia/checks.json) · [Projection illustration](geometry/projection_illustration.json)",
    "",
    "## Figures",
    "",
]
figures = sorted((OUT / "figures").glob("*.pdf")) if (OUT / "figures").exists() else []
if figures:
    for f in figures:
        lines.append(f"- [{f.name}](figures/{f.name})")
else:
    lines.append(
        "The checks are complete. Run `python run.py figures` with the same output directory to generate the figures."
    )
movies = sorted((OUT / "movies").glob("*.gif"))
if movies:
    lines += [
        "",
        "## Supporting movies",
        "",
        "Observed exposures with retained annotations; no temporal interpolation or model refitting.",
        "",
    ]
    for movie in movies:
        item = f"- [{movie.stem} — GIF](movies/{movie.name})"
        mp4 = movie.with_suffix(".mp4")
        if mp4.exists():
            item += f" · [MP4](movies/{mp4.name})"
        lines.append(item)
    lines += [
        "",
        "Caption and input details: docs/ANIMATIONS.md in the source package.",
    ]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("Readable run report: " + str(OUT / "SUMMARY.md"))
