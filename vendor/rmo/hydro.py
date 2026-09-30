from __future__ import annotations

import math

from scipy.optimize import brentq

from .models import Solution, State, Wave


def is_hydrodynamic(left: State, right: State, tol: float = 1.0e-14) -> bool:
    return max(abs(x) for x in (*left.B, *right.B, left.u[1], left.u[2], right.u[1], right.u[2])) <= tol


def vacuum_condition(left: State, right: State, gamma: float) -> bool:
    if not is_hydrodynamic(left, right):
        return False
    a_left = math.sqrt(gamma * left.p / left.rho)
    a_right = math.sqrt(gamma * right.p / right.rho)
    return right.u[0] - left.u[0] >= 2.0 * (a_left + a_right) / (gamma - 1.0)


def _pressure_function(pstar: float, state: State, gamma: float) -> tuple[float, float]:
    a = math.sqrt(gamma * state.p / state.rho)
    if pstar > state.p:
        A = 2.0 / ((gamma + 1.0) * state.rho)
        B = (gamma - 1.0) * state.p / (gamma + 1.0)
        root = math.sqrt(A / (pstar + B))
        f = (pstar - state.p) * root
        derivative = root * (1.0 - 0.5 * (pstar - state.p) / (pstar + B))
        return f, derivative
    exponent = (gamma - 1.0) / (2.0 * gamma)
    ratio = pstar / state.p
    f = 2.0 * a / (gamma - 1.0) * (ratio**exponent - 1.0)
    derivative = (1.0 / (state.rho * a)) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, derivative


def exact_hydro_riemann(left: State, right: State, gamma: float, policy: str) -> Solution:
    if vacuum_condition(left, right, gamma):
        raise ValueError("VACUUM_FORMED")

    def residual(pstar: float) -> float:
        return _pressure_function(pstar, left, gamma)[0] + _pressure_function(pstar, right, gamma)[0] + right.u[0] - left.u[0]

    low = 1.0e-14 * min(left.p, right.p)
    high = max(left.p, right.p, 1.0)
    while residual(high) < 0.0:
        high *= 2.0
        if high > 1.0e6 * max(left.p, right.p, 1.0):
            raise RuntimeError("SEARCH_DOMAIN_TRUNCATED")
    pstar = float(brentq(residual, low, high, xtol=1.0e-13, rtol=1.0e-14))
    f_left = _pressure_function(pstar, left, gamma)[0]
    f_right = _pressure_function(pstar, right, gamma)[0]
    ustar = 0.5 * (left.u[0] + right.u[0] + f_right - f_left)

    def star_density(state: State) -> float:
        if pstar > state.p:
            ratio = pstar / state.p
            g = (gamma - 1.0) / (gamma + 1.0)
            return state.rho * (ratio + g) / (g * ratio + 1.0)
        return state.rho * (pstar / state.p) ** (1.0 / gamma)

    lstar = State(star_density(left), pstar, (ustar, 0.0, 0.0), (0.0, 0.0, 0.0), id="X1", role="intermediate")
    rstar = State(star_density(right), pstar, (ustar, 0.0, 0.0), (0.0, 0.0, 0.0), id="X2", role="intermediate")
    waves: list[Wave] = []
    if pstar > left.p:
        a = math.sqrt(gamma * left.p / left.rho)
        speed = left.u[0] - a * math.sqrt((gamma + 1.0) * pstar / (2.0 * gamma * left.p) + (gamma - 1.0) / (2.0 * gamma))
        waves.append(Wave(0, "shock", "fast_minus", speed, left, lstar, {"pstar": pstar}))
    else:
        a_left = math.sqrt(gamma * left.p / left.rho)
        a_star = math.sqrt(gamma * pstar / lstar.rho)
        waves.append(Wave(0, "rarefaction", "fast_minus", (left.u[0] - a_left, ustar - a_star), left, lstar, {"isentropic": True}))
    waves.append(Wave(1, "contact", "entropy", ustar, lstar, rstar, {"pressure_continuous": True, "velocity_continuous": True}))
    if pstar > right.p:
        a = math.sqrt(gamma * right.p / right.rho)
        speed = right.u[0] + a * math.sqrt((gamma + 1.0) * pstar / (2.0 * gamma * right.p) + (gamma - 1.0) / (2.0 * gamma))
        waves.append(Wave(2, "shock", "fast_plus", speed, rstar, right, {"pstar": pstar}))
    else:
        a_right = math.sqrt(gamma * right.p / right.rho)
        a_star = math.sqrt(gamma * pstar / rstar.rho)
        waves.append(Wave(2, "rarefaction", "fast_plus", (ustar + a_star, right.u[0] + a_right), rstar, right, {"isentropic": True}))
    return Solution("hydro_exact_0", policy, waves, [lstar, rstar], ["OK"], {"method": "exact_euler_pressure_match", "pstar": pstar, "ustar": ustar})
