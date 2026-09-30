"""Independent SI flux, entropy and characteristic checks; fixed mapping to AIA."""

from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
import os

P = Path(os.environ["FRONT_OUTPUT"]) / "fan"
cfg = json.loads((P / "MODEL_PROTOCOL.json").read_text())
G = 5 / 3
mu = 4 * np.pi * 1e-7
R = cfg["surface_radius_km"]
deg_km = R * np.pi / 180


def st(row):
    return {
        "id": row["id"],
        "r": float(row["rho_kg_m3"]),
        "p": float(row["p_Pa"]),
        "u": np.array([row[k] for k in ["un_km_s", "ut1_km_s", "ut2_km_s"]], float)
        * 1e3,
        "B": np.array([row[k] for k in ["Bn_G", "Bt1_G", "Bt2_G"]], float) * 1e-4,
    }


def chars(s):
    aa = G * s["p"] / s["r"]
    bb = s["B"] @ s["B"] / (mu * s["r"])
    bn = s["B"][0] ** 2 / (mu * s["r"])
    disc = np.sqrt(max(0, (aa + bb) ** 2 - 4 * aa * bn))
    cf = np.sqrt(((aa + bb) + disc) / 2)
    cs = np.sqrt(aa * bn / cf**2)
    ca = np.sqrt(bn)
    return s["u"][0] + np.array([-cf, -ca, -cs, 0, cs, ca, cf])


def flux(s, D):
    u = s["u"].copy()
    u[0] -= D
    r = s["r"]
    p = s["p"]
    B = s["B"]
    b = B @ B
    E = p / (G - 1) + 0.5 * r * (u @ u) + b / (2 * mu)
    return np.r_[
        r * u[0],
        r * u[0] * u + np.array([p + b / (2 * mu), 0, 0]) - B[0] * B / mu,
        u[0] * B[1:] - B[0] * u[1:],
        u[0] * (E + p + b / (2 * mu)) - B[0] * (u @ B) / mu,
    ]


allchecks = []
fluxrows = []
stateinputs = []
predictions = []
comparisons = []
age = []
obs = pd.read_csv(P / "inputs/display_feature_primary.csv")
tt = pd.to_datetime(obs.UTC, utc=True)
dt = (tt - tt.iloc[0]).dt.total_seconds().to_numpy()
thetafirst = obs.peak_angle_deg.iloc[0]
for cid in ["P080", "P100", "P120"]:
    data = json.loads((P / "outputs" / f"{cid}_solution.json").read_text())
    df = pd.read_csv(P / "outputs" / f"{cid}_states.csv")
    S = [st(r) for r in df.to_dict("records")]
    W = pd.read_csv(P / "outputs" / f"{cid}_waves.csv")
    v0 = 750000.0
    r0 = max(x["r"] for x in S)
    p0 = max(x["p"] for x in S)
    B0 = max(np.linalg.norm(x["B"]) for x in S)
    scales = np.array(
        [r0 * v0]
        + [r0 * v0 * v0 + p0 + B0 * B0 / (2 * mu)] * 3
        + [v0 * B0] * 2
        + [v0 * (G * p0 / (G - 1) + r0 * v0 * v0 / 2 + B0 * B0 / mu)]
    )
    one = []
    for k, w in W.iterrows():
        if w.structure == "rarefaction":
            continue
        A, B = S[k : k + 2]
        D = w.speed_left_km_s * 1000
        fa, fb = flux(A, D), flux(B, D)
        res = np.abs(fa - fb) / scales
        for quantity, l, r, z, norm in zip(
            [
                "mass",
                "normal_momentum",
                "t1_momentum",
                "t2_momentum",
                "t1_induction",
                "t2_induction",
                "energy",
            ],
            fa,
            fb,
            res,
            scales,
        ):
            fluxrows.append(
                {
                    "case": cid,
                    "family": w.family,
                    "quantity": quantity,
                    "LEFT_flux": l,
                    "RIGHT_flux": r,
                    "scale_SI": norm,
                    "normalized_residual": z,
                }
            )
        check = {
            "family": w.family,
            "structure": w.structure,
            "max_scaled_flux_residual": float(res.max()),
            "Bn_jump_G": float((B["B"][0] - A["B"][0]) * 1e4),
        }
        if w.structure == "shock":
            up, dn = (A, B) if fa[0] > 0 else (B, A)
            entropy = np.log(dn["p"] / up["p"]) - G * np.log(dn["r"] / up["r"])
            el, er = chars(A), chars(B)
            marg = [el[k] - D, D - er[k]]
            if k > 0:
                marg.append(D - el[k - 1])
            if k < 6:
                marg.append(er[k + 1] - D)
            check.update(
                entropy_increase_over_cv=float(entropy),
                min_Lax_margin_km_s=float(min(marg) / 1000),
                compression=float(dn["r"] / up["r"]),
            )
        elif w.structure == "contact":
            check.update(
                density_ratio_RIGHT_LEFT=float(B["r"] / A["r"]),
                max_material_speed_error_km_s=float(
                    max(abs(A["u"][0] - D), abs(B["u"][0] - D)) / 1000
                ),
                pressure_jump_Pa=float(B["p"] - A["p"]),
                B_jump_G=float(np.max(abs(B["B"] - A["B"])) * 1e4),
                velocity_jump_km_s=float(np.max(abs(B["u"] - A["u"])) / 1000),
            )
        elif w.structure == "rotation":
            sign = np.sign(A["u"][0] - D)
            resw = (
                B["u"][1:]
                - A["u"][1:]
                - sign * (B["B"][1:] - A["B"][1:]) / np.sqrt(mu * A["r"])
            )
            check.update(
                Walen_residual_km_s=float(np.max(abs(resw)) / 1000),
                relative_density_jump=float(B["r"] / A["r"] - 1),
                relative_pressure_jump=float(B["p"] / A["p"] - 1),
                Bt_norm_ratio=float(
                    np.linalg.norm(B["B"][1:]) / np.linalg.norm(A["B"][1:])
                ),
            )
        one.append(check)
    rare = data["rarefaction_samples_normalized"]
    ra = []
    for fam in sorted(set(x["family"] for x in rare)):
        vals = [x for x in rare if x["family"] == fam]
        ss = []
        for x in vals:
            ss.append(
                {
                    "r": x["rho"],
                    "p": x["p"],
                    "u": np.array(x["u"]),
                    "B": np.array(x["B"]) * np.sqrt(mu),
                }
            )
        idx = {"fast_minus": 0, "slow_minus": 2, "slow_plus": 4, "fast_plus": 6}[fam]
        eigen = np.array([chars(x)[idx] for x in ss])
        ent = np.array([x["p"] / x["r"] ** G for x in ss])
        direction = -1 if fam.endswith("minus") else 1
        ra.append(
            {
                "family": fam,
                "samples": len(ss),
                "isentrope_relative_error": float(np.max(abs(ent / ent[0] - 1))),
                "characteristic_monotonic_in_spatial_order": bool(
                    np.all(-direction * np.diff(eigen) >= -1e-9)
                ),
                "density_tail_over_head": float(ss[-1]["r"] / ss[0]["r"]),
                "pressure_tail_over_head": float(ss[-1]["p"] / ss[0]["p"]),
            }
        )
    ordered = bool(
        np.all(
            W.speed_left_km_s.to_numpy()[1:] - W.speed_right_km_s.to_numpy()[:-1] > 0
        )
    )
    # Explicitly include the recorded contact/rotation/positivity diagnostics in
    # the aggregate flag. Use the existing protocol tolerances and normalization.
    tol = cfg["tolerances"]["contact_matching"]
    norm = data["normalization"]
    vn = norm["velocity_m_s"] / 1000
    pn = norm["pressure_Pa"]
    bn = norm["magnetic_field_T"] * 1e4
    gates = {
        "positive_states": all(x["r"] > 0 and x["p"] > 0 for x in S),
        "common_normal_B": max(abs(x["B"][0] - S[0]["B"][0]) for x in S) / (bn * 1e-4)
        < tol,
        "matching": data["matching_residual"] < tol,
        "fluxes": all(
            x["max_scaled_flux_residual"] < cfg["tolerances"]["independent_scaled_flux"]
            for x in one
        ),
        "shock_entropy_and_Lax": all(
            x.get("entropy_increase_over_cv", 1) > 0
            and x.get("min_Lax_margin_km_s", 1) > 0
            and x.get("compression", 2) > 1
            for x in one
        ),
        "contacts": all(
            max(
                x["max_material_speed_error_km_s"] / vn,
                abs(x["pressure_jump_Pa"]) / pn,
                x["B_jump_G"] / bn,
                x["velocity_jump_km_s"] / vn,
            )
            < tol
            for x in one
            if x["structure"] == "contact"
        ),
        "rotations": all(
            max(
                x["Walen_residual_km_s"] / vn,
                abs(x["relative_density_jump"]),
                abs(x["relative_pressure_jump"]),
                abs(x["Bt_norm_ratio"] - 1),
            )
            < tol
            for x in one
            if x["structure"] == "rotation"
        ),
        "rarefactions": all(
            x["isentrope_relative_error"] < 1e-9
            and x["characteristic_monotonic_in_spatial_order"]
            and 0 < x["density_tail_over_head"] < 1
            and 0 < x["pressure_tail_over_head"] < 1
            for x in ra
        ),
        "wave_order": ordered,
    }
    passed = bool(data["solver_valid"] and all(gates.values()))
    out = {
        "case": cid,
        "independent_checks_pass": passed,
        "states_positive": all(x["r"] > 0 and x["p"] > 0 for x in S),
        "ordered": ordered,
        "discontinuities": one,
        "rarefactions": ra,
        "max_scaled_flux_residual": max(c["max_scaled_flux_residual"] for c in one),
        "initial_left_temperature_MK": float(df.T_MK.iloc[0]),
        "fast_shock_speed_km_s": float(W.speed_left_km_s.iloc[-1]),
        "contact_speed_km_s": float(W.speed_left_km_s.iloc[3]),
        "rarefaction_bounds_km_s": W.iloc[0][
            ["speed_left_km_s", "speed_right_km_s"]
        ].to_list(),
    }
    if cid == "P100":
        prior = pd.read_csv(P / "inputs/reference_states.csv")
        columns = [x for x in prior.columns if x != "id"]
        out["baseline_max_scaled_state_difference"] = float(
            np.max(
                np.abs(df[columns].to_numpy() - prior[columns].to_numpy())
                / np.maximum(1.0, np.abs(prior[columns].to_numpy()))
            )
        )
    out["independent_condition_flags"] = {
        key: bool(value) for key, value in gates.items()
    }
    allchecks.append(out)
    for i, side in [(0, "LEFT"), (-1, "RIGHT")]:
        for key in [
            "ne_cm3",
            "p_Pa",
            "T_MK",
            "un_km_s",
            "ut1_km_s",
            "ut2_km_s",
            "Bn_G",
            "Bt1_G",
            "Bt2_G",
        ]:
            stateinputs.append(
                {
                    "case": cid,
                    "side": side,
                    "quantity": key,
                    "value": df.iloc[i][key],
                    "provenance_class": "ASSUMED",
                    "source": "Prescribed reference model; pressure factor as declared in MODEL_PROTOCOL.json",
                    "limitation": "Not measured in the AIA sector",
                }
            )
    Dfast = float(W.speed_left_km_s.iloc[-1])
    theta0 = thetafirst - Dfast * 120 / deg_km
    for i in range(5):
        for k, w in W.iterrows():
            predictions.append(
                {
                    "case": cid,
                    "UTC": obs.UTC.iloc[i],
                    "model_age_s": 120 + dt[i],
                    "interface_theta_deg": theta0,
                    "family": w.family,
                    "structure": w.structure,
                    "theta_left_deg": theta0
                    + w.speed_left_km_s * (120 + dt[i]) / deg_km,
                    "theta_right_deg": theta0
                    + w.speed_right_km_s * (120 + dt[i]) / deg_km,
                }
            )
        pred = thetafirst + Dfast * dt[i] / deg_km
        comparisons.append(
            {
                "case": cid,
                "UTC": obs.UTC.iloc[i],
                "elapsed_s": dt[i],
                "observed_display_peak_deg": obs.peak_angle_deg.iloc[i],
                "predicted_conditional_fast_deg": pred,
                "model_minus_display_peak_deg": pred - obs.peak_angle_deg.iloc[i],
                "used_as_anchor": i == 0,
            }
        )
    for tau0 in [60, 120, 240]:
        origin = thetafirst - Dfast * tau0 / deg_km
        for k, w in W.iterrows():
            age.append(
                {
                    "case": cid,
                    "initial_age_s": tau0,
                    "family": w.family,
                    "interface_theta_deg": origin,
                    "theta_left_last_deg": origin
                    + w.speed_left_km_s * (tau0 + dt[-1]) / deg_km,
                    "theta_right_last_deg": origin
                    + w.speed_right_km_s * (tau0 + dt[-1]) / deg_km,
                }
            )
pd.DataFrame(fluxrows).to_csv(P / "independent_fluxes.csv", index=False)
pd.DataFrame(stateinputs).to_csv(P / "input_provenance.csv", index=False)
pd.DataFrame(predictions).to_csv(P / "predicted_positions.csv", index=False)
pd.DataFrame(comparisons).to_csv(P / "kinematic_comparison.csv", index=False)
pd.DataFrame(age).to_csv(P / "age_sensitivity.csv", index=False)
(P / "independent_model_checks.json").write_text(json.dumps(allchecks, indent=2) + "\n")
summary = {
    "all_three_cases_pass": all(x["independent_checks_pass"] for x in allchecks),
    "max_scaled_flux_residual": max(x["max_scaled_flux_residual"] for x in allchecks),
    "cases": [
        {
            k: x[k]
            for k in [
                "case",
                "initial_left_temperature_MK",
                "fast_shock_speed_km_s",
                "contact_speed_km_s",
                "rarefaction_bounds_km_s",
            ]
        }
        for x in allchecks
    ],
    "baseline_later_frame_residuals_deg": [
        x["model_minus_display_peak_deg"]
        for x in comparisons
        if x["case"] == "P100" and not x["used_as_anchor"]
    ],
    "baseline_last_frame_age_sensitivity": [
        x
        for x in age
        if x["case"] == "P100" and x["family"] in ["fast_minus", "entropy"]
    ],
    "geometry": "Local planar fan mapped to an assumed spherical great-circle coordinate; not a spherical MHD solution",
    "physical_identification": "No observed contact or rarefaction identified",
}
(P / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

if not summary["all_three_cases_pass"]:
    raise RuntimeError("Independent physical checks failed.")
