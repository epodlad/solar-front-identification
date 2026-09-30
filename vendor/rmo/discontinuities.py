from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .models import Solution, State, Wave
from .physics import CONSERVATION_TOL, scaled_inf_residual, states_close, total_pressure


def _named(state: State, state_id: str, role: str) -> State:
    return replace(state, id=state_id, role=role)


def recognize_constant(left: State, right: State, policy: str) -> Solution | None:
    if not states_close(left, right):
        return None
    return Solution(
        "constant_0",
        policy,
        [],
        [],
        ["OK"],
        {"method": "exact_state_equality", "zero_strength_waves_recorded": 7},
    )


def recognize_contact(left: State, right: State, gamma: float, policy: str) -> Solution | None:
    if abs(left.B[0]) <= 1.0e-12:
        return None
    same_u = np.allclose(left.uvec, right.uvec, rtol=5.0e-9, atol=5.0e-9)
    same_b = np.allclose(left.bvec, right.bvec, rtol=5.0e-9, atol=5.0e-9)
    same_p = math.isclose(left.p, right.p, rel_tol=5.0e-9, abs_tol=5.0e-9)
    if not (same_u and same_b and same_p):
        return None
    speed = left.u[0]
    resid = scaled_inf_residual(left, right, speed, gamma)
    if resid > CONSERVATION_TOL:
        return None
    lstate = _named(left, "X_L0", "initial_left")
    rstate = _named(right, "X_R0", "initial_right")
    wave = Wave(0, "contact", "entropy", speed, lstate, rstate, {"rh_scaled_inf": resid})
    return Solution("contact_0", policy, [wave], [], ["OK"], {"method": "exact_contact_invariants"})


def recognize_tangential(left: State, right: State, gamma: float, policy: str) -> Solution | None:
    if max(abs(left.B[0]), abs(right.B[0])) > 1.0e-12:
        return None
    if not math.isclose(left.u[0], right.u[0], rel_tol=5.0e-9, abs_tol=5.0e-9):
        return None
    if not math.isclose(total_pressure(left), total_pressure(right), rel_tol=5.0e-9, abs_tol=5.0e-9):
        return None
    speed = left.u[0]
    resid = scaled_inf_residual(left, right, speed, gamma)
    if resid > CONSERVATION_TOL:
        return None
    lstate = _named(left, "X_L0", "initial_left")
    rstate = _named(right, "X_R0", "initial_right")
    wave = Wave(0, "tangential", "entropy", speed, lstate, rstate, {"rh_scaled_inf": resid, "total_pressure": total_pressure(left)})
    return Solution("tangential_0", policy, [wave], [], ["OK"], {"method": "exact_tangential_invariants"})


def recognize_rotation(left: State, right: State, gamma: float, policy: str) -> Solution | None:
    scalar_equal = all(
        math.isclose(a, b, rel_tol=5.0e-9, abs_tol=5.0e-9)
        for a, b in [(left.rho, right.rho), (left.p, right.p), (left.u[0], right.u[0]), (left.B[0], right.B[0])]
    )
    bt_norm_equal = math.isclose(np.linalg.norm(left.bvec[1:]), np.linalg.norm(right.bvec[1:]), rel_tol=5.0e-9, abs_tol=5.0e-9)
    if not (scalar_equal and bt_norm_equal and abs(left.B[0]) > 1.0e-12):
        return None
    ca = abs(left.B[0]) / math.sqrt(left.rho)
    candidates = [(left.u[0] - ca, "alfven_minus"), (left.u[0] + ca, "alfven_plus")]
    best_speed, best_family = min(candidates, key=lambda item: scaled_inf_residual(left, right, item[0], gamma))
    resid = scaled_inf_residual(left, right, best_speed, gamma)
    if resid > CONSERVATION_TOL:
        return None
    lstate = _named(left, "X_L0", "initial_left")
    rstate = _named(right, "X_R0", "initial_right")
    wave = Wave(0, "rotation", best_family, best_speed, lstate, rstate, {"rh_scaled_inf": resid, "thermodynamics_continuous": True})
    return Solution("rotation_0", policy, [wave], [], ["OK"], {"method": "exact_rotational_invariants"})
