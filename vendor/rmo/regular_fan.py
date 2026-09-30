from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np
from scipy.optimize import least_squares

from .models import Solution, State, Wave
from .physics import scaled_inf_residual
from .topology import classify_geometry
from .wave_curves import CurveResult, apply_rotation, solve_magnetosonic_to_pressure


@dataclass(frozen=True)
class RegularFanSearch:
    solutions: tuple[Solution, ...]
    rejected: tuple[dict[str, object], ...]
    starts_attempted: int
    root_set_stable: bool
    complete: bool
    domain_codes: tuple[str, ...]


def _state_vector(state: State) -> np.ndarray:
    return np.asarray([state.rho, state.p, *state.u, state.B[1], state.B[2]], dtype=float)


def _solution_distance(first: Solution, second: Solution) -> float:
    """Scaled distance between two complete regular fans.

    State-space clustering avoids treating rotation angles that differ by
    2*pi as different physical roots.
    """

    first_states = [first.waves[0].left_state] + [wave.right_state for wave in first.waves]
    second_states = [second.waves[0].left_state] + [wave.right_state for wave in second.waves]
    if len(first_states) != len(second_states):
        return math.inf
    distances: list[float] = []
    for one, two in zip(first_states, second_states):
        a, b = _state_vector(one), _state_vector(two)
        distances.append(float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b))))))
    return max(distances, default=0.0)


@dataclass(frozen=True)
class _Regions:
    left: tuple[CurveResult, CurveResult, CurveResult]
    right_from_outer: tuple[CurveResult, CurveResult, CurveResult]

    @property
    def left_contact(self) -> State:
        return self.left[-1].downstream

    @property
    def right_contact(self) -> State:
        return self.right_from_outer[-1].downstream


def _build_regions(left: State, right: State, gamma: float, params: np.ndarray, deadline: float | None = None) -> _Regions:
    p_left_fast, angle_left, p_star, angle_right, p_right_fast = (
        math.exp(float(params[0])), float(params[1]), math.exp(float(params[2])),
        float(params[3]), math.exp(float(params[4])),
    )
    left_fast = solve_magnetosonic_to_pressure(left, p_left_fast, gamma, "fast", -1, search_starts=False, deadline=deadline)
    left_rotation = apply_rotation(left_fast.downstream, angle_left, -1, gamma)
    left_slow = solve_magnetosonic_to_pressure(left_rotation.downstream, p_star, gamma, "slow", -1, search_starts=False, deadline=deadline)

    right_fast = solve_magnetosonic_to_pressure(right, p_right_fast, gamma, "fast", 1, search_starts=False, deadline=deadline)
    right_rotation = apply_rotation(right_fast.downstream, angle_right, 1, gamma)
    right_slow = solve_magnetosonic_to_pressure(right_rotation.downstream, p_star, gamma, "slow", 1, search_starts=False, deadline=deadline)
    return _Regions((left_fast, left_rotation, left_slow), (right_fast, right_rotation, right_slow))


def _contact_residual(left: State, right: State, gamma: float, params: np.ndarray, deadline: float | None = None) -> np.ndarray:
    try:
        regions = _build_regions(left, right, gamma, params, deadline)
        one, two = regions.left_contact, regions.right_contact
        scale = np.maximum(1.0, np.maximum(np.abs([*one.u, one.B[1], one.B[2]]), np.abs([*two.u, two.B[1], two.B[2]])))
        return (np.array([*one.u, one.B[1], one.B[2]]) - np.array([*two.u, two.B[1], two.B[2]])) / scale
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(5, 1.0e4)


def _starts(left: State, right: State, count: int, seed: int) -> list[np.ndarray]:
    base_pressure = max(1.0e-12, 0.5 * (left.p + right.p))
    starts = [np.array([math.log(left.p * 1.2), 0.0, math.log(base_pressure), 0.0, math.log(right.p * 1.2)])]
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        pressure_factors = np.exp(rng.uniform(-1.5, 1.5, size=3))
        starts.append(np.array([
            math.log(left.p * pressure_factors[0]),
            rng.uniform(-math.pi, math.pi),
            math.log(base_pressure * pressure_factors[1]),
            rng.uniform(-math.pi, math.pi),
            math.log(right.p * pressure_factors[2]),
        ]))
    return starts


def _wave_speed_bounds(result: CurveResult) -> tuple[float, float]:
    if isinstance(result.speed, tuple):
        return result.speed
    return float(result.speed), float(result.speed)


def _solution_from_regions(regions: _Regions, left: State, right: State, gamma: float, index: int, params: np.ndarray, residual: float) -> Solution:
    states: list[State] = [replace(left, id="X_L0", role="initial_left")]
    for number, result in enumerate(regions.left, start=1):
        states.append(replace(result.downstream, id=f"X_L{number}", role="intermediate"))
    right_outer = replace(right, id="X_R0", role="initial_right")
    right_inward = [replace(result.downstream, id=f"X_R{number}", role="intermediate") for number, result in enumerate(regions.right_from_outer, start=1)]
    contact_right = right_inward[-1]

    waves: list[Wave] = []
    families = ("fast_minus", "alfven_minus", "slow_minus")
    for order, (result, family) in enumerate(zip(regions.left, families)):
        waves.append(Wave(order, result.structure, family, result.speed, states[order], states[order + 1], result.checks))
    contact_speed = states[-1].u[0]
    contact_error = scaled_inf_residual(states[-1], contact_right, contact_speed, gamma)
    waves.append(Wave(3, "contact", "entropy", contact_speed, states[-1], contact_right, {"rh_scaled_inf": contact_error}))

    # Right curves were generated outer-to-inner, so reverse their order and state orientation.
    inner_to_outer = [regions.right_from_outer[2], regions.right_from_outer[1], regions.right_from_outer[0]]
    right_states = [contact_right, right_inward[1], right_inward[0], right_outer]
    right_families = ("slow_plus", "alfven_plus", "fast_plus")
    for offset, (result, family) in enumerate(zip(inner_to_outer, right_families), start=4):
        waves.append(Wave(offset, result.structure, family, result.speed, right_states[offset - 4], right_states[offset - 3], result.checks))

    return Solution(
        id=f"regular_fan_{index}", policy="REGULAR_EVOLUTIONARY_1.0", waves=waves,
        intermediate_states=states[1:] + list(reversed(right_inward)), domain_codes=["OK"],
        root_provenance={
            "method": "five_parameter_contact_matching",
            "parameters": [float(value) for value in params],
            "contact_scaled_inf": residual,
            "source": "Torrilhon-2002; Takahashi-Yamada-2014",
        },
    )


def _valid_solution(solution: Solution) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    speeds = [_wave_speed_bounds_from_wave(wave) for wave in solution.waves]
    for first, second in zip(speeds[:-1], speeds[1:]):
        if first[1] > second[0] + 1.0e-9 * max(1.0, abs(first[1]), abs(second[0])):
            reasons.append("wave_speed_order")
            break
    for wave in solution.waves:
        if wave.structure in {"shock", "rotation", "contact"} and float(wave.checks.get("rh_scaled_inf", 0.0)) > 1.0e-10:
            reasons.append(f"rh:{wave.order}")
        if wave.structure == "shock" and not bool(wave.checks.get("entropy_admissible", True)):
            reasons.append(f"entropy:{wave.order}")
        if wave.structure == "rarefaction" and not bool(wave.checks.get("characteristic_speed_monotonic", False)):
            reasons.append(f"rarefaction_monotonicity:{wave.order}")
    if float(solution.root_provenance["contact_scaled_inf"]) > 1.0e-10:
        reasons.append("contact_matching")
    return not reasons, reasons


def _wave_speed_bounds_from_wave(wave: Wave) -> tuple[float, float]:
    if isinstance(wave.speed, tuple):
        return float(wave.speed[0]), float(wave.speed[1])
    return float(wave.speed), float(wave.speed)


def search_regular_fans(
    left: State,
    right: State,
    gamma: float,
    *,
    starts: int = 33,
    seed: int = 24090501,
    wall_time_seconds: float = 1800.0,
) -> RegularFanSearch:
    geometry = classify_geometry(left, right)
    if geometry.bn_zero or geometry.left_bt_zero or geometry.right_bt_zero:
        return RegularFanSearch((), (), 0, False, False, ("ODE_EVENT_DEGENERACY", "BRANCH_ENUMERATION_INCOMPLETE"))
    reference_pressure = max(left.p, right.p)
    lower = np.array([math.log(reference_pressure * 1.0e-6), -2.0 * math.pi, math.log(reference_pressure * 1.0e-6), -2.0 * math.pi, math.log(reference_pressure * 1.0e-6)])
    upper = np.array([math.log(reference_pressure * 1.0e6), 2.0 * math.pi, math.log(reference_pressure * 1.0e6), 2.0 * math.pi, math.log(reference_pressure * 1.0e6)])
    accepted: list[Solution] = []
    rejected: list[dict[str, object]] = []
    start_time = time.monotonic()
    deadline = start_time + wall_time_seconds
    attempted = 0
    for start in _starts(left, right, starts, seed):
        if time.monotonic() - start_time > wall_time_seconds:
            return RegularFanSearch(tuple(accepted), tuple(rejected), attempted, False, False, ("SEARCH_DOMAIN_TRUNCATED", "ROOT_SET_NOT_STABLE"))
        attempted += 1
        fitted = least_squares(
            lambda value: _contact_residual(left, right, gamma, value, deadline),
            np.minimum(np.maximum(start, lower + 1.0e-12), upper - 1.0e-12),
            bounds=(lower, upper), method="trf", max_nfev=300,
            xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10,
        )
        if time.monotonic() > deadline:
            return RegularFanSearch(tuple(accepted), tuple(rejected), attempted, False, False, ("SEARCH_DOMAIN_TRUNCATED", "ROOT_SET_NOT_STABLE"))
        residual = float(np.max(np.abs(_contact_residual(left, right, gamma, fitted.x, deadline))))
        if not fitted.success or residual > 1.0e-8:
            rejected.append({"start": attempted - 1, "failed_predicate": "ROOT_NOT_CONVERGED", "residual": residual, "message": fitted.message})
            continue
        try:
            regions = _build_regions(left, right, gamma, fitted.x, deadline)
            solution = _solution_from_regions(regions, left, right, gamma, len(accepted), fitted.x, residual)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rejected.append({"start": attempted - 1, "failed_predicate": type(exc).__name__, "message": str(exc)})
            continue
        valid, reasons = _valid_solution(solution)
        if not valid:
            rejected.append({"start": attempted - 1, "failed_predicate": "candidate_validation", "reasons": reasons, "candidate": solution.serializable()})
            continue
        if any(_solution_distance(solution,old) <= 1e-8 for old in accepted):
            rejected.append({"start": attempted - 1, "failed_predicate": "root_cluster_duplicate", "candidate_id": solution.id})
            continue
        accepted.append(solution)
    # A fixed start count alone cannot prove stability; the caller must compare
    # against a doubled search before setting this flag.
    return RegularFanSearch(tuple(accepted), tuple(rejected), attempted, False, False, ("ROOT_SET_NOT_STABLE",))


def search_regular_fans_stabilized(
    left: State,
    right: State,
    gamma: float,
    *,
    base_starts: int = 33,
    seed: int = 24090501,
    wall_time_seconds: float = 1800.0,
    cluster_tolerance: float = 1.0e-8,
) -> RegularFanSearch:
    """Compare the frozen multistart search with a doubled search.

    A stable finite root set is an operational numerical criterion, not a
    mathematical proof that isolated roots do not exist.  Any truncation or
    mismatch therefore remains fail-closed.
    """

    started = time.monotonic()
    base = search_regular_fans(
        left, right, gamma, starts=base_starts, seed=seed,
        wall_time_seconds=wall_time_seconds,
    )
    remaining = wall_time_seconds - (time.monotonic() - started)
    if remaining <= 0.0 or "SEARCH_DOMAIN_TRUNCATED" in base.domain_codes:
        return RegularFanSearch(
            base.solutions, base.rejected, base.starts_attempted,
            False, False, ("SEARCH_DOMAIN_TRUNCATED", "ROOT_SET_NOT_STABLE"),
        )
    doubled = search_regular_fans(
        left, right, gamma, starts=2 * base_starts, seed=seed,
        wall_time_seconds=remaining,
    )
    stable = (
        bool(base.solutions)
        and "SEARCH_DOMAIN_TRUNCATED" not in doubled.domain_codes
        and len(base.solutions) == len(doubled.solutions)
        and all(any(_solution_distance(one, two) <= cluster_tolerance for two in doubled.solutions) for one in base.solutions)
        and all(any(_solution_distance(one, two) <= cluster_tolerance for one in base.solutions) for two in doubled.solutions)
    )
    rejected = tuple(base.rejected) + tuple(
        {**item, "search": "doubled"} for item in doubled.rejected
    )
    if not stable:
        codes = ["ROOT_SET_NOT_STABLE"]
        if "SEARCH_DOMAIN_TRUNCATED" in doubled.domain_codes:
            codes.insert(0, "SEARCH_DOMAIN_TRUNCATED")
        return RegularFanSearch(
            doubled.solutions, rejected,
            base.starts_attempted + doubled.starts_attempted,
            False, False, tuple(codes),
        )
    solutions: list[Solution] = []
    for index, solution in enumerate(doubled.solutions):
        provenance = dict(solution.root_provenance)
        provenance.update({
            "root_set_stability": "same_clusters_after_doubled_search",
            "base_starts": base_starts,
            "doubled_starts": 2 * base_starts,
            "cluster_tolerance": cluster_tolerance,
        })
        solutions.append(replace(solution, id=f"regular_fan_{index}", root_provenance=provenance))
    return RegularFanSearch(
        tuple(solutions), rejected,
        base.starts_attempted + doubled.starts_attempted,
        True, False, ("BRANCH_ENUMERATION_INCOMPLETE",),
    )
