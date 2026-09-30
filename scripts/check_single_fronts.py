"""Independent algebraic tests of the three complete neighboring state pairs.

Inputs use rho0, U0, rho0*U0**2, and sqrt(mu0*rho0*U0**2).
The normalized magnetic permeability is one. All velocities are in the
same local normal/tangential basis; only the normal frame boost is applied.
"""

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["FRONT_OUTPUT"]) / "single_fronts"
OUT.mkdir(parents=True, exist_ok=True)


def flux(state, speed, gamma):
    """Seven scalar conserved fluxes, evaluated in the front rest frame."""
    rho, p = state["rho"], state["p"]
    w = [state["u"][0] - speed, *state["u"][1:]]
    b = state["B"]
    b2 = sum(x * x for x in b)
    energy = p / (gamma - 1) + rho * sum(x * x for x in w) / 2 + b2 / 2
    return [
        rho * w[0],
        rho * w[0] ** 2 + p + b2 / 2 - b[0] ** 2,
        rho * w[0] * w[1] - b[0] * b[1],
        rho * w[0] * w[2] - b[0] * b[2],
        w[0] * b[1] - b[0] * w[1],
        w[0] * b[2] - b[0] * w[2],
        w[0] * (energy + p + b2 / 2) - b[0] * sum(x * y for x, y in zip(w, b)),
    ]


def speeds(state, gamma):
    """Slow, normal Alfven and fast speeds relative to the local plasma."""
    a2 = gamma * state["p"] / state["rho"]
    va2 = sum(x * x for x in state["B"]) / state["rho"]
    can2 = state["B"][0] ** 2 / state["rho"]
    disc = math.sqrt(max(0.0, (a2 + va2) ** 2 - 4 * a2 * can2))
    cf2 = (a2 + va2 + disc) / 2
    return dict(
        slow=math.sqrt(a2 * can2 / cf2), alfven_n=math.sqrt(can2), fast=math.sqrt(cf2)
    )


results = []
for folder, stem, expected in [
    ("E11", "E11_F1", "fast"),
    ("E05", "E05_FAST", "fast"),
    ("E05", "E05_SLOW", "slow"),
]:
    q = json.loads((ROOT / f"data/{folder}/{stem}_normalized_input.json").read_text())
    si = json.loads((ROOT / f"data/{folder}/{stem}_SI_states.json").read_text())
    unit = (
        si["states"]["UPSTREAM"]["u_shock_frame_m_s"][0]
        if folder == "E11"
        else si["scales_SI"]["U0_m_s"]
    ) / 1000
    left, right = q["left"], q["right"]
    gamma, speed = q["gamma"], q["front_speed"]
    fl, fr = flux(left, speed, gamma), flux(right, speed, gamma)
    up, dn = (left, right) if fl[0] > 0 else (right, left)
    c1, c2 = speeds(up, gamma), speeds(dn, gamma)
    w1, w2 = abs(up["u"][0] - speed), abs(dn["u"][0] - speed)
    compression = dn["rho"] / up["rho"]
    entropy = math.log(dn["p"] / up["p"]) - gamma * math.log(compression)
    fast = w1 > c1["fast"] and c2["alfven_n"] < w2 < c2["fast"]
    slow = c1["slow"] < w1 < c1["alfven_n"] and w2 < c2["slow"]
    residual = max(abs(x - y) for x, y in zip(fl, fr))
    gates = {
        "positive_states": all(s["rho"] > 0 and s["p"] > 0 for s in (left, right)),
        "normal_field_continuity": left["B"][0] == right["B"][0],
        "seven_fluxes": residual < 1e-12,
        "compression": compression > 1,
        "entropy_increase": entropy > 0,
        "expected_family": fast if expected == "fast" else slow,
    }
    result = dict(
        case=stem,
        label=f"{folder}: {expected}",
        family=expected,
        flux_LEFT=fl,
        flux_RIGHT=fr,
        max_normalized_flux_residual=residual,
        entropy_increase_over_cv=entropy,
        compression=compression,
        upstream_side="LEFT" if fl[0] > 0 else "RIGHT",
        checks=gates,
    )
    for label, chars, flow in [("upstream", c1, w1), ("downstream", c2, w2)]:
        result[label] = {name + "_km_s": value * unit for name, value in chars.items()}
        result[label]["flow_km_s"] = flow * unit
    results.append(result)
    if not all(gates.values()):
        raise RuntimeError(f"{stem}: failed conditions: {gates}")
(OUT / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
for r in results:
    print(
        f"{r['case']}: {r['family']} shock; flux residual {r['max_normalized_flux_residual']:.2e}"
    )
