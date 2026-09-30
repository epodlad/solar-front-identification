"""Align MAG with PAS acquisition windows and calculate HT and Walen fits.

Adapted from the independent calculation; legacy comparison files removed.
No boundary normal or complete discontinuity classification is inferred.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import numpy as np
import cdflib

MU0 = 4 * np.pi * 1e-7  # Conventional SI value used throughout the calculations
MP = 1.67262192369e-27
EV = 1.602176634e-19
START = np.datetime64("2021-08-30T10:21:24", "ns")
END = np.datetime64("2021-08-30T10:21:28", "ns")
EXPECTED = {
    "solo_L2_mag-rtn-normal_20210830_V03.cdf": "1f5edb8e5102eaac4e21130684a14d4ca15b4d03cd6f61ff37cb29cc016d1fa8",
    "solo_L2_swa-pas-grnd-mom_20210830_V03.cdf": "59a42214b6e1596e45987087b316fe0525038fcbecf3c6d7e9ec7d33249ca400",
}


def safe_json(x):
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in x]
    if isinstance(x, (np.datetime64, np.timedelta64)):
        return str(x)
    if isinstance(x, np.generic):
        return safe_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def write_json(path, obj):
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def valid(cdf, name, arr):
    at = cdf.varattsget(name)
    ok = np.isfinite(arr)
    for key, compare in [("VALIDMIN", np.greater_equal), ("VALIDMAX", np.less_equal)]:
        if key in at:
            ok &= compare(arr, at[key])
    if "FILLVAL" in at and np.isfinite(at["FILLVAL"]):
        ok &= arr != at["FILLVAL"]
    return bool(np.all(ok))


def ols(x, y):
    # One pooled intercept. The 3N components are not independent acquisitions.
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    xc, yc = x - x.mean(), y - y.mean()
    xx, yy, xy = xc @ xc, yc @ yc, xc @ yc
    slope = xy / xx
    intercept = y.mean() - slope * x.mean()
    return dict(
        slope=float(slope),
        intercept_km_s=float(intercept),
        pooled_r=float(xy / np.sqrt(xx * yy)),
        residual_rms_km_s=float(np.sqrt(np.mean((y - slope * x - intercept) ** 2))),
    )


def ht_matrix(b, v):
    # Both inputs float64, B in nT, V in km/s. Common B unit cancels.
    k = np.sum(b * b, axis=1)[:, None, None] * np.eye(3) - b[:, :, None] * b[:, None, :]
    a = np.sum(k, axis=0)
    rhs = np.sum(np.einsum("nij,nj->ni", k, v), axis=0)
    ht = np.linalg.solve(a, rhs)
    u = v - ht
    # (km/s)*nT = 1e-6 V/m = 1e-3 mV/m.
    e0 = -np.cross(v, b) * 1e-3
    eht = -np.cross(u, b) * 1e-3
    evary = -np.cross(v - v.mean(axis=0), b) * 1e-3
    d0, dht, dc = [np.mean(np.sum(q * q, axis=1)) for q in [e0, eht, evary]]
    return ht, dict(
        condition_K=float(np.linalg.cond(a)),
        eigenvalues_K_nT2=np.linalg.eigvalsh(a),
        relative_normal_equation_residual=float(
            np.linalg.norm(a @ ht - rhs) / np.linalg.norm(rhs)
        ),
        E_HT_vector_rms_mV_m=float(np.sqrt(dht)),
        E_input_frame_vector_rms_mV_m=float(np.sqrt(d0)),
        E_mean_velocity_frame_vector_rms_mV_m=float(np.sqrt(dc)),
        D_HT_over_D_input=float(dht / d0),
        D_HT_over_D_mean_velocity=float(dht / dc),
        electric_field_note="Inferred convection field -V cross B; not an independent electric-field measurement.",
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = a.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = args.source
    hashes = {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in EXPECTED
    }
    assert (
        hashes == EXPECTED
    ), "CDF checksum differs from the version used in the article"
    mc = cdflib.CDF(raw / list(EXPECTED)[0])
    pc = cdflib.CDF(raw / list(EXPECTED)[1])
    for c, name, unit in [
        (mc, "B_RTN", "nT"),
        (pc, "V_RTN", "km/s"),
        (pc, "N", "particles cm^-3"),
        (pc, "T", "eV"),
        (pc, "Half_interval", "s"),
    ]:
        assert c.varattsget(name)["UNITS"] == unit
    assert pc.varattsget("Epoch")["DELTA_PLUS_VAR"] == "Half_interval"
    assert pc.varattsget("Epoch")["DELTA_MINUS_VAR"] == "Half_interval"
    mt = cdflib.cdfepoch.to_datetime(mc.varget("EPOCH"))
    pt = cdflib.cdfepoch.to_datetime(pc.varget("Epoch"))
    lo, hi = np.datetime64("2021-08-30T10:18"), np.datetime64("2021-08-30T10:23")
    ms, ps = (mt >= lo) & (mt < hi), (pt >= lo) & (pt < hi)
    mt, pt = mt[ms], pt[ps]
    mag_names = ["B_RTN", "QUALITY_FLAG", "QUALITY_BITMASK", "VECTOR_TIME_RESOLUTION"]
    pas_names = [
        "V_RTN",
        "N",
        "T",
        "Half_interval",
        "quality_factor",
        "Info",
        "unrecovered_count",
        "total_count",
        "V_SOLO_RTN",
    ]
    mag = {v: mc.varget(v)[ms] for v in mag_names}
    pas = {v: pc.varget(v)[ps] for v in pas_names}
    flags = {f"MAG/{v}": valid(mc, v, mag[v]) for v in mag_names}
    flags.update({f"PAS/{v}": valid(pc, v, pas[v]) for v in pas_names})
    assert all(flags.values()), flags
    assert np.all(pas["N"] > 0) and np.all(pas["Half_interval"] > 0)
    assert np.all(np.diff(mt) > np.timedelta64(0, "ns")) and np.all(
        np.diff(pt) > np.timedelta64(0, "ns")
    )
    metadata = {
        "sha256": hashes,
        "MAG_attributes": {v: mc.varattsget(v) for v in ["EPOCH"] + mag_names},
        "PAS_attributes": {v: pc.varattsget(v) for v in ["Epoch"] + pas_names},
        "MAG_global_attributes": mc.globalattsget(),
        "PAS_global_attributes": pc.globalattsget(),
        "valid_range_and_fill_checks_5min": flags,
        "quality_policy": "Retain all finite, valid-range samples; expose flags. No undocumented PAS quality threshold imposed.",
    }
    write_json(out / "cdf_metadata_and_quality.json", metadata)
    h = np.rint(pas["Half_interval"].astype(float) * 1e9).astype("timedelta64[ns]")
    begin, finish = pt - h, pt + h
    b64, b32, counts, rows = [], [], [], []
    for i, (t0, t1) in enumerate(zip(begin, finish)):
        take = (mt >= t0) & (mt < t1)
        assert take.sum() == 8, (pt[i], take.sum())
        b64.append(mag["B_RTN"][take].mean(axis=0, dtype=np.float64))
        b32.append(mag["B_RTN"][take].mean(axis=0, dtype=np.float32))
        counts.append(int(take.sum()))
        relation = (
            "before"
            if t1 <= START
            else (
                "after"
                if t0 >= END
                else (
                    "inside_core" if t0 >= START and t1 <= END else "overlaps_core_edge"
                )
            )
        )
        r = dict(
            time_UTC=str(pt[i]),
            acquisition_start_UTC=str(t0),
            acquisition_end_exclusive_UTC=str(t1),
            relation_by_full_exposure=relation,
            MAG_samples=int(take.sum()),
            first_MAG_UTC=str(mt[take][0]),
            last_MAG_UTC=str(mt[take][-1]),
            N_cm3=float(pas["N"][i]),
            T_proton_eV=float(pas["T"][i]),
            p_proton_Pa=float(pas["N"][i]) * 1e6 * float(pas["T"][i]) * EV,
            PAS_quality_factor=float(pas["quality_factor"][i]),
            PAS_Info=int(pas["Info"][i]),
            PAS_unrecovered_count=float(pas["unrecovered_count"][i]),
            MAG_quality_flags=";".join(map(str, np.unique(mag["QUALITY_FLAG"][take]))),
            MAG_quality_bitmasks=";".join(
                map(str, np.unique(mag["QUALITY_BITMASK"][take]))
            ),
        )
        for j, c in enumerate("RTN"):
            r[f"B_{c}_nT"] = float(b64[-1][j])
            r[f"V_{c}_km_s"] = float(pas["V_RTN"][i, j])
        rows.append(r)
    b64, b32 = np.array(b64), np.array(b32)
    v, n = pas["V_RTN"].astype(float), pas["N"].astype(float)
    write_csv(out / "all_5min_aligned_samples.csv", rows)
    np.savez_compressed(
        out / "aligned_5min.npz",
        time=pt,
        acquisition_start=begin,
        acquisition_end=finish,
        B=b64,
        V=v,
        N=n,
        T=pas["T"].astype(float),
        B_native_float32_mean=b32,
        raw_MAG_time=mt,
        raw_MAG_B=mag["B_RTN"],
    )
    comparisons = []
    detailed = {}
    primary = None
    for pad in [10, 15, 20]:
        take = (pt >= START - np.timedelta64(pad, "s")) & (
            pt <= END + np.timedelta64(pad, "s")
        )
        b, vv, nn = b64[take], v[take], n[take]
        ht, htd = ht_matrix(b, vv)
        va = b * 1e-9 / np.sqrt(MU0 * MP * nn[:, None] * 1e6) / 1000
        u = vv - ht
        fit = ols(va, u)
        rawsel = (mt >= START - np.timedelta64(pad, "s")) & (
            mt <= END + np.timedelta64(pad, "s")
        )
        bb = mag["B_RTN"][rawsel].astype(float)
        dc = bb - bb.mean(axis=0)
        eig = np.linalg.eigvalsh(dc.T @ dc / (len(bb) - 1))
        rec = dict(
            padding_s=pad,
            PAS_moments=int(take.sum()),
            **fit,
            V_HT_R_km_s=float(ht[0]),
            V_HT_T_km_s=float(ht[1]),
            V_HT_N_km_s=float(ht[2]),
            condition_K=htd["condition_K"],
            E_HT_vector_rms_mV_m=htd["E_HT_vector_rms_mV_m"],
            D_HT_over_D_input=htd["D_HT_over_D_input"],
            D_HT_over_D_mean_velocity=htd["D_HT_over_D_mean_velocity"],
            plain_MVA_middle_min=float(eig[1] / eig[0]),
        )
        comparisons.append(rec)
        detailed[str(pad)] = {
            "HT": htd,
            "MAG_quality_flags_in_window": np.unique(mag["QUALITY_FLAG"][rawsel]),
            "MAG_quality_bitmasks_in_window": np.unique(mag["QUALITY_BITMASK"][rawsel]),
            "component_OLS": {c: ols(va[:, j], u[:, j]) for j, c in enumerate("RTN")},
            "slope_through_origin": float(np.sum(va * u) / np.sum(va * va)),
            "full_exposure_relations": {
                r: sum(
                    rows[i]["relation_by_full_exposure"] == r for i in np.where(take)[0]
                )
                for r in ["before", "inside_core", "overlaps_core_edge", "after"]
            },
        }
        if pad == 15:
            primary = dict(take=take, ht=ht, va=va, u=u)
            write_csv(
                out / "primary_15s_samples.csv", [rows[i] for i in np.where(take)[0]]
            )
    write_csv(out / "window_comparison.csv", comparisons)
    means = []
    for side, mask in [("before", finish <= START), ("after", begin >= END)]:
        s = primary["take"] & mask
        indices = np.where(s)[0]
        r = dict(
            side=side,
            PAS_moments=int(s.sum()),
            first_exposure_start_UTC=str(begin[indices[0]]),
            last_exposure_end_UTC=str(finish[indices[-1]]),
            N_cm3=float(n[s].mean()),
            T_proton_eV=float(pas["T"][s].mean(dtype=np.float64)),
            mean_p_proton_Pa=float(
                np.mean(n[s] * 1e6 * pas["T"][s].astype(float) * EV)
            ),
            mean_B_magnitude_nT=float(np.linalg.norm(b64[s], axis=1).mean()),
        )
        for j, c in enumerate("RTN"):
            r[f"B_{c}_nT"] = float(b64[s, j].mean())
            r[f"V_{c}_km_s"] = float(v[s, j].mean())
        means.append(r)
    write_csv(out / "before_after_descriptive_means.csv", means)
    summary = dict(
        event="Suen et al. 2023 Event 2 CS2",
        core_UTC=[str(START), str(END)],
        source_DOI="10.1051/0004-6361/202345922",
        published_slope=0.973,
        independent_implementation="K matrix normal equations and centered scalar OLS",
        basis="RTN; supplied CDF V_RTN velocities, followed only by a fitted constant HT boost",
        numeric_precision="float64 after CDF read",
        MAG_samples_5min=len(mt),
        PAS_moments_5min=len(pt),
        PAS_acquisition_duration_s=np.unique(2 * pas["Half_interval"]),
        PAS_median_cadence_s=float(np.median(np.diff(pt) / np.timedelta64(1, "s"))),
        MAG_cadence_s=float(np.median(np.diff(mt) / np.timedelta64(1, "s"))),
        core_midpoint_PAS_moments=int(np.sum((pt >= START) & (pt <= END))),
        core_overlapping_exposures=[
            rows[i] for i in np.where((finish > START) & (begin < END))[0]
        ],
        MAG_quality_flags=np.unique(mag["QUALITY_FLAG"]),
        MAG_quality_bitmasks=np.unique(mag["QUALITY_BITMASK"]),
        PAS_quality_factor_range=[
            float(pas["quality_factor"].min()),
            float(pas["quality_factor"].max()),
        ],
        comparisons=comparisons,
        diagnostics=detailed,
        conclusion="Positive Alfvenic relation reproduced. Full independent RD or shock classification not established.",
        limitations=[
            "One 1-s acquisition in the 4-s core; proton cadence is approximately 4 s.",
            "Pooled RTN components are not 3N independent temporal measurements; no OLS standard errors reported.",
            "Window sensitivity is not a confidence interval. The primary window remains +/-15 s.",
            "Plain MVA middle/min ratio is not the hybrid-MVA max/middle criterion used by Suen et al.",
            "Density is proton-only and isotropic Walen relation is assumed; no total pressure inferred from proton T.",
            "Before/after means are descriptive context, not fitted RH states or independently established upstream/downstream.",
            "Exact paper processing and calibration version have not been reproduced.",
        ],
        environment={
            "python": sys.version,
            "numpy": np.__version__,
            "cdflib": cdflib.__version__,
            "platform": platform.platform(),
        },
    )
    write_json(out / "checks.json", summary)
    print(
        json.dumps(
            safe_json(
                {
                    "comparisons": comparisons,
                    "core_exposures": summary["core_overlapping_exposures"],
                    "means": means,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
