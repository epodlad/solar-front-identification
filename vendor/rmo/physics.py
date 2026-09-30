from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import State


SPECIFICATION_ID = "RMO-R6A-SPEC-1.0.0"
RHO_P_MIN = 1.0e-12
BN_TOL = 1.0e-12
CONSERVATION_TOL = 1.0e-10
ENTROPY_ALLOWANCE = -1.0e-10


def validate_state(state: State, gamma: float) -> list[str]:
    values: Iterable[float] = (state.rho, state.p, *state.u, *state.B, gamma)
    if not all(math.isfinite(float(x)) for x in values):
        return ["INVALID_STATE"]
    if gamma <= 1.0 or state.rho <= 0.0 or state.p <= 0.0:
        return ["INVALID_STATE"]
    if state.rho <= RHO_P_MIN or state.p <= RHO_P_MIN:
        return ["OUTSIDE_NUMERIC_DOMAIN"]
    return ["OK"]


def validate_pair(left: State, right: State, gamma: float) -> list[str]:
    lc = validate_state(left, gamma)
    rc = validate_state(right, gamma)
    if lc != ["OK"]:
        return lc
    if rc != ["OK"]:
        return rc
    scale = max(1.0, abs(left.B[0]), abs(right.B[0]))
    if abs(left.B[0] - right.B[0]) > BN_TOL * scale:
        return ["DIVERGENCE_CONSTRAINT_FAIL"]
    return ["OK"]


def total_pressure(state: State) -> float:
    return state.p + 0.5 * float(np.dot(state.bvec, state.bvec))


def total_energy(state: State, gamma: float) -> float:
    return (
        state.p / (gamma - 1.0)
        + 0.5 * state.rho * float(np.dot(state.uvec, state.uvec))
        + 0.5 * float(np.dot(state.bvec, state.bvec))
    )


def conservative(state: State, gamma: float) -> np.ndarray:
    return np.array(
        [
            state.rho,
            state.rho * state.u[0],
            state.rho * state.u[1],
            state.rho * state.u[2],
            state.B[1],
            state.B[2],
            total_energy(state, gamma),
        ],
        dtype=float,
    )


def flux(state: State, gamma: float) -> np.ndarray:
    rho, un, ut1, ut2 = state.rho, *state.u
    bn, bt1, bt2 = state.B
    ptotal = total_pressure(state)
    energy = total_energy(state, gamma)
    udotb = float(np.dot(state.uvec, state.bvec))
    return np.array(
        [
            rho * un,
            rho * un * un + ptotal - bn * bn,
            rho * un * ut1 - bn * bt1,
            rho * un * ut2 - bn * bt2,
            un * bt1 - ut1 * bn,
            un * bt2 - ut2 * bn,
            (energy + ptotal) * un - bn * udotb,
        ],
        dtype=float,
    )


def rh_vector(left: State, right: State, speed: float, gamma: float) -> np.ndarray:
    return flux(right, gamma) - flux(left, gamma) - speed * (
        conservative(right, gamma) - conservative(left, gamma)
    )


def scaled_inf_residual(left: State, right: State, speed: float, gamma: float) -> float:
    resid = rh_vector(left, right, speed, gamma)
    scale = np.maximum(
        1.0,
        np.maximum(np.abs(flux(left, gamma)), np.abs(flux(right, gamma))),
    )
    return float(np.max(np.abs(resid) / scale))


def characteristic_speeds(state: State, gamma: float) -> dict[str, float | list[float]]:
    a2 = gamma * state.p / state.rho
    b2_over_rho = float(np.dot(state.bvec, state.bvec)) / state.rho
    ca2 = state.B[0] ** 2 / state.rho
    disc = max(0.0, (a2 + b2_over_rho) ** 2 - 4.0 * a2 * ca2)
    root = math.sqrt(disc)
    cf2 = max(0.0, 0.5 * (a2 + b2_over_rho + root))
    # Subtractive-cancellation-safe slow speed.
    cs2 = 0.0 if cf2 == 0.0 else max(0.0, a2 * ca2 / cf2)
    cf, cs, ca = math.sqrt(cf2), math.sqrt(cs2), math.sqrt(ca2)
    un = state.u[0]
    eigenvalues = [un - cf, un - ca, un - cs, un, un + cs, un + ca, un + cf]
    return {"sound": math.sqrt(a2), "slow": cs, "alfven_n": ca, "fast": cf, "eigenvalues": eigenvalues}


def entropy_over_cv(state: State, gamma: float) -> float:
    return math.log(state.p) - gamma * math.log(state.rho)


def states_close(left: State, right: State, rtol: float = 5.0e-9) -> bool:
    a = np.array([left.rho, left.p, *left.u, *left.B], dtype=float)
    b = np.array([right.rho, right.p, *right.u, *right.B], dtype=float)
    return bool(np.allclose(a, b, rtol=rtol, atol=rtol))
