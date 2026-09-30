from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .forward import uniform_slab_observer
from .models import State
from .physics import scaled_inf_residual, validate_pair, validate_state
from .rh import solve_shock_branches_with_provenance
from .solver import RiemannSolver
from .uncertainty import calibration_fixture


def _state(value: dict[str, Any], role: str = "intermediate") -> State:
    return State.from_mapping(value, role=role)


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def run_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tol = manifest["tolerances"]
    solver = RiemannSolver()
    results: list[dict[str, Any]] = []

    for case in manifest["benchmarks"]:
        case_id = case["id"]
        gamma = float(case.get("gamma", manifest["default_gamma"]))
        expected = case["expected"]
        details: dict[str, Any] = {}
        passed = False
        blocker: str | None = None

        if case_id == "B00_constant":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            passed = result.complete and len(result.solutions) == 1 and len(result.solutions[0].waves) == 0
            details = result.serializable()
        elif case_id == "B01_contact":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            wave = result.solutions[0].waves[0] if result.solutions else None
            passed = bool(result.complete and wave and wave.structure == "contact" and _close(float(wave.speed), expected["speed"], tol["exact_value_relative"]))
            details = result.serializable()
        elif case_id == "B02_rotation_right":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            wave = result.solutions[0].waves[0] if result.solutions else None
            passed = bool(result.complete and wave and wave.structure == "rotation" and wave.family == expected["family"] and _close(float(wave.speed), expected["speed"], tol["exact_value_relative"]) and wave.checks["rh_scaled_inf"] <= tol["conservation_scaled_inf"])
            details = result.serializable()
        elif case_id == "B03_hydro_M3_stationary":
            upstream = _state(case["upstream"], "upstream")
            downstream = _state(case["downstream"], "downstream")
            residual = scaled_inf_residual(upstream, downstream, expected["shock_speed"], gamma)
            compression = downstream.rho / upstream.rho
            pressure_ratio = downstream.p / upstream.p
            passed = residual <= tol["conservation_scaled_inf"] and _close(compression, expected["compression"], tol["exact_value_relative"]) and _close(pressure_ratio, expected["pressure_ratio"], tol["exact_value_relative"])
            details = {"scaled_residual": residual, "compression": compression, "pressure_ratio": pressure_ratio}
        elif case_id in {"B04_oblique_fast_RH", "B05_oblique_slow_RH"}:
            upstream = _state(case["upstream"], "upstream")
            branches, root_provenance = solve_shock_branches_with_provenance(upstream, gamma)
            doubled, doubled_provenance = solve_shock_branches_with_provenance(upstream, gamma, samples=8193)
            family = expected["family"]
            matching = [b for b in branches if b.family == family and b.evolutionary]
            stable_roots = len(branches) == len(doubled) and all(
                any(old.family == new.family and abs(old.compression - new.compression) <= tol["root_cluster_scaled_distance"] for new in doubled)
                for old in branches
            )
            if not matching:
                blocker = "NO_EXPECTED_SHOCK_BRANCH"
                details = {"branches": [b.__dict__ for b in branches], "root_provenance": root_provenance}
            else:
                branch = min(matching, key=lambda item: abs(item.compression - expected["compression"]))
                scalar_checks = [
                    _close(branch.compression, expected["compression"], 5.0e-9),
                    branch.scaled_residual <= tol["conservation_scaled_inf"],
                    stable_roots,
                    len([b for b in branches if b.evolutionary]) == 1,
                ]
                if case_id == "B05_oblique_slow_RH":
                    scalar_checks.extend(
                        [
                            _close(branch.downstream.p / upstream.p, expected["pressure_ratio"], 5.0e-5),
                            _close(abs(branch.downstream.B[1] / upstream.B[1]), expected["abs_bt_ratio"], 5.0e-5),
                            _close(branch.entropy_change_over_cv, expected["entropy_over_cv"], 5.0e-5),
                        ]
                    )
                passed = all(scalar_checks)
                details = {
                    "compression": branch.compression,
                    "family": branch.family,
                    "evolutionary": branch.evolutionary,
                    "pressure_ratio": branch.downstream.p / upstream.p,
                    "abs_bt_ratio": abs(branch.downstream.B[1] / upstream.B[1]),
                    "entropy_over_cv": branch.entropy_change_over_cv,
                    "scaled_residual": branch.scaled_residual,
                    "downstream": branch.downstream.serializable(),
                    "stable_after_doubled_scan": stable_roots,
                    "root_provenance": root_provenance,
                    "doubled_scan_provenance": doubled_provenance,
                }
        elif case_id == "B06_Sod":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            provenance = result.solutions[0].root_provenance if result.solutions else {}
            wave_order = [wave.structure for wave in result.solutions[0].waves] if result.solutions else []
            passed = result.complete and _close(provenance.get("pstar", math.nan), expected["p_star"], tol["exact_value_relative"]) and _close(provenance.get("ustar", math.nan), expected["u_star"], tol["exact_value_relative"]) and wave_order == ["rarefaction", "contact", "shock"]
            details = result.serializable()
        elif case_id == "B07_tangential_Bn0":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            wave = result.solutions[0].waves[0] if result.solutions else None
            passed = bool(result.complete and wave and wave.structure == "tangential" and wave.checks["rh_scaled_inf"] <= tol["conservation_scaled_inf"])
            details = result.serializable()
        elif case_id == "B08_Bn_mismatch":
            codes = validate_pair(_state(case["left"]), _state(case["right"]), gamma)
            passed = codes == [expected["domain_code"]]
            details = {"domain_codes": codes, "solve_attempted": False}
        elif case_id == "B09_negative_pressure":
            codes = validate_state(_state(case["left"]), gamma)
            passed = codes == [expected["domain_code"]]
            details = {"domain_codes": codes, "solve_attempted": False}
        elif case_id == "B10_vacuum_expansion":
            result = solver.solve(_state(case["left"]), _state(case["right"]), gamma, "REGULAR_EVOLUTIONARY_1.0")
            passed = result.domain_codes == [expected["domain_code"]] and result.complete
            details = result.serializable()
        elif case_id == "B11_Brio_Wu_policy":
            policy_results = {
                policy: solver.solve(_state(case["left"]), _state(case["right"]), gamma, policy).serializable()
                for policy in expected["policies_required"]
            }
            # Recognition of policy dependence is necessary but not sufficient:
            # the frozen suite forbids passing without enumerated expected fans.
            passed = False
            blocker = "BRANCH_ENUMERATION_INCOMPLETE"
            details = {"policy_results": policy_results, "policy_dependence_recognized": True}
        elif case_id == "B12_uniform_slab_forward":
            prediction = uniform_slab_observer(**case["plasma"])
            passed = _close(prediction["optically_thin_intensity"], expected["optically_thin_intensity"], tol["exact_value_relative"]) and _close(prediction["centroid_A"], expected["centroid_A"], tol["exact_value_relative"])
            details = prediction
        elif case_id == "B13_uncertainty_calibration":
            calibration = calibration_fixture(case["replicates_min"], manifest["seed"])
            passed = bool(calibration["pass"])
            details = calibration
        else:
            blocker = "UNKNOWN_BENCHMARK"

        if not passed and blocker is None:
            blocker = "REFERENCE_DISAGREEMENT"
        results.append({"id": case_id, "pass": passed, "blocker": blocker, "details": details})

    passed_count = sum(int(item["pass"]) for item in results)
    failed = [item["id"] for item in results if not item["pass"]]
    return {
        "specification_id": manifest["specification_id"],
        "manifest_id": manifest["manifest_id"],
        "benchmark_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(failed),
        "failed": failed,
        "suite_pass": not failed,
        "r6b_conformance": "PASS" if not failed else "FAIL_CLOSED",
        "independent_cross_code": "NOT_RUN_REFERENCE_IMPLEMENTATION_UNAVAILABLE",
        "results": results,
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
