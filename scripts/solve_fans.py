"""Construct the three prescribed fans on the retained regular branch.

Adapted from the archived calculation: paths and failure reporting changed;
wave curves, starting states and numerical settings are unchanged.
"""

from pathlib import Path
from dataclasses import replace
import sys, json, time, math, hashlib
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import os

ROOT = Path(__file__).resolve().parents[1]
P = Path(os.environ["FRONT_OUTPUT"]) / "fan"
sys.path.insert(0, str(ROOT / "vendor"))
from rmo.models import State
from rmo.regular_fan import (
    _contact_residual,
    _build_regions,
    _solution_from_regions,
    _valid_solution,
)
from rmo.wave_curves import solve_rarefaction_to_pressure

cfg = json.loads((P / "MODEL_PROTOCOL.json").read_text())
ref = json.loads((P / "inputs/reference_model/left_right.json").read_text())
norm = ref["normalization"]
G = ref["gamma"]
V = norm["velocity_m_s"] / 1000


def state(val, label):
    r, p, un, u1, u2, bn, b1, b2 = val
    return State(r, p, (un, u1, u2), (bn, b1, b2), id=label)


baseL = state(ref["states"]["LEFT"]["primitive_normalized"], "LEFT")
baseR = state(ref["states"]["RIGHT"]["primitive_normalized"], "RIGHT")
s = pd.read_csv(P / "inputs/reference_states.csv").set_index("id")
angle = lambda key: np.arctan2(s.loc[key, "Bt2_G"], s.loc[key, "Bt1_G"])
warm = np.array(
    [
        np.log(s.loc["X_L1", "p_Pa"] / norm["pressure_Pa"]),
        angle("X_L2") - angle("X_L1"),
        np.log(s.loc["X_L3", "p_Pa"] / norm["pressure_Pa"]),
        angle("X_R2") - angle("X_R1"),
        np.log(s.loc["X_R1", "p_Pa"] / norm["pressure_Pa"]),
    ]
)
(P / "outputs").mkdir(exist_ok=True)
out = []
for case in [cfg["new_cases"][1], cfg["new_cases"][0], cfg["new_cases"][2]]:
    L = replace(baseL, p=baseL.p * case["left_pressure_factor"])
    R = baseR
    start = time.monotonic()
    deadline = start + cfg["tolerances"]["wall_time_seconds_per_case"]
    data = {
        "id": case["id"],
        "left_pressure_factor": case["left_pressure_factor"],
        "LEFT": L.serializable(),
        "RIGHT": R.serializable(),
        "normalization": norm,
        "gamma": G,
        "all_input_plasma_states": "ASSUMED",
        "branch_enumeration_complete": False,
        "root_set_stability_tested": False,
    }
    try:
        pressure = max(L.p, R.p)
        lo = np.array(
            [
                np.log(pressure * 1e-6),
                -2 * np.pi,
                np.log(pressure * 1e-6),
                -2 * np.pi,
                np.log(pressure * 1e-6),
            ]
        )
        hi = np.array(
            [
                np.log(pressure * 1e6),
                2 * np.pi,
                np.log(pressure * 1e6),
                2 * np.pi,
                np.log(pressure * 1e6),
            ]
        )
        fit = least_squares(
            lambda x: _contact_residual(L, R, G, x, deadline),
            warm,
            bounds=(lo, hi),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=100,
        )
        err = float(np.max(abs(_contact_residual(L, R, G, fit.x, deadline))))
        regions = _build_regions(L, R, G, fit.x, deadline)
        sol = _solution_from_regions(regions, L, R, G, 0, fit.x, err)
        valid, reasons = _valid_solution(sol)
        data.update(
            {
                "optimization_success": bool(fit.success),
                "optimizer_message": str(fit.message),
                "nfev": int(fit.nfev),
                "matching_residual": err,
                "solver_valid": bool(valid),
                "solver_rejections": reasons,
                "solution": sol.serializable(),
            }
        )
        allstates = [sol.waves[0].left_state] + [w.right_state for w in sol.waves]
        rows = []
        for st in allstates:
            rho = st.rho * norm["density_kg_m3"]
            p = st.p * norm["pressure_Pa"]
            ne = rho / 1.67262192369e-27 / 1e6
            rows.append(
                {
                    "id": st.id,
                    "rho_kg_m3": rho,
                    "ne_cm3": ne,
                    "p_Pa": p,
                    "T_MK": p / (2 * ne * 1e6 * 1.380649e-23) / 1e6,
                    "un_km_s": st.u[0] * V,
                    "ut1_km_s": st.u[1] * V,
                    "ut2_km_s": st.u[2] * V,
                    "Bn_G": st.B[0] * norm["magnetic_field_T"] * 1e4,
                    "Bt1_G": st.B[1] * norm["magnetic_field_T"] * 1e4,
                    "Bt2_G": st.B[2] * norm["magnetic_field_T"] * 1e4,
                }
            )
        pd.DataFrame(rows).to_csv(
            P / "outputs" / f'{case["id"]}_states.csv', index=False
        )
        wr = []
        rare = []
        for w in sol.waves:
            speed = np.atleast_1d(w.speed) * V
            wr.append(
                {
                    "family": w.family,
                    "structure": w.structure,
                    "speed_left_km_s": float(speed.min()),
                    "speed_right_km_s": float(speed.max()),
                }
            )
            if w.structure == "rarefaction":
                up = w.left_state if w.family.endswith("minus") else w.right_state
                down = w.right_state if w.family.endswith("minus") else w.left_state
                direc = -1 if w.family.endswith("minus") else 1
                for p in np.linspace(up.p, down.p, 41):
                    rr = solve_rarefaction_to_pressure(
                        up, float(p), G, w.family.split("_")[0], direc
                    )
                    st = rr.downstream
                    rare.append(
                        {
                            "family": w.family,
                            "rho": st.rho,
                            "p": st.p,
                            "u": list(st.u),
                            "B": list(st.B),
                        }
                    )
        pd.DataFrame(wr).to_csv(P / "outputs" / f'{case["id"]}_waves.csv', index=False)
        data["rarefaction_samples_normalized"] = rare
    except Exception as exc:
        data.update(
            {"solver_valid": False, "error": type(exc).__name__ + ": " + str(exc)}
        )
    data["wall_seconds"] = time.monotonic() - start
    (P / "outputs" / f'{case["id"]}_solution.json').write_text(
        json.dumps(data, indent=2) + "\n"
    )
    out.append(
        {
            k: data.get(k)
            for k in [
                "id",
                "solver_valid",
                "matching_residual",
                "wall_seconds",
                "error",
            ]
        }
    )
    print(json.dumps(out[-1]), flush=True)
(P / "solver_run_summary.json").write_text(json.dumps(out, indent=2) + "\n")

if not all(
    x["solver_valid"] and x["matching_residual"] < cfg["tolerances"]["contact_matching"]
    for x in out
):
    raise RuntimeError("One or more fans failed construction or contact matching.")
