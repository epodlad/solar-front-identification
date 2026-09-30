from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .models import State
from .physics import (
    characteristic_speeds,
    conservative,
    entropy_over_cv,
    flux,
    scaled_inf_residual,
)


@dataclass(frozen=True)
class CurveResult:
    downstream: State
    structure: str
    family: str
    speed: float | tuple[float, float]
    checks: dict[str, object]


def primitive_vector(state: State) -> np.ndarray:
    return np.array([state.rho, *state.u, state.B[1], state.B[2], state.p], dtype=float)


def state_from_primitive(vector: np.ndarray, bn: float) -> State:
    return State(
        rho=float(vector[0]), p=float(vector[6]),
        u=(float(vector[1]), float(vector[2]), float(vector[3])),
        B=(bn, float(vector[4]), float(vector[5])),
    )


def _central_jacobian(function: object, vector: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    output = np.zeros((7, 7), dtype=float)
    for column in range(7):
        delta = step * max(1.0, abs(float(vector[column])))
        perturbation = np.zeros(7)
        perturbation[column] = delta
        output[:, column] = (function(vector + perturbation) - function(vector - perturbation)) / (2.0 * delta)
    return output


def primitive_eigensystem(state: State, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    vector = primitive_vector(state)
    bn = state.B[0]

    def q(value: np.ndarray) -> np.ndarray:
        return conservative(state_from_primitive(value, bn), gamma)

    def f(value: np.ndarray) -> np.ndarray:
        return flux(state_from_primitive(value, bn), gamma)

    dq = _central_jacobian(q, vector)
    df = _central_jacobian(f, vector)
    matrix = np.linalg.solve(dq, df)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if np.max(np.abs(np.imag(eigenvalues))) > 1.0e-8:
        raise RuntimeError("ODE_EVENT_DEGENERACY")
    return np.real(eigenvalues), np.real(eigenvectors)


def solve_rarefaction_to_pressure(
    upstream: State,
    pressure_downstream: float,
    gamma: float,
    family: str,
    direction: int,
    deadline: float | None = None,
) -> CurveResult:
    if family not in {"fast", "slow"} or direction not in {-1, 1}:
        raise ValueError("invalid rarefaction family or direction")
    if not (1.0e-12 < pressure_downstream <= upstream.p):
        raise ValueError("rarefaction requires 0 < p_downstream <= p_upstream")
    entropy_constant = upstream.p / upstream.rho**gamma
    initial = np.array([*upstream.u, upstream.B[1], upstream.B[2]], dtype=float)
    bn = upstream.B[0]

    def rhs(pressure: float, vector: np.ndarray) -> np.ndarray:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("SEARCH_DOMAIN_TRUNCATED")
        rho = (pressure / entropy_constant) ** (1.0 / gamma)
        state = State(rho, pressure, tuple(vector[:3]), (bn, float(vector[3]), float(vector[4])))
        return analytic_pressure_tangent(state, gamma, family, direction)

    integrated = solve_ivp(
        rhs, (upstream.p, pressure_downstream), initial,
        method="DOP853", rtol=1.0e-10, atol=1.0e-12,
    )
    if not integrated.success:
        raise RuntimeError("ROOT_NOT_CONVERGED")
    final = integrated.y[:, -1]
    downstream = State(
        (pressure_downstream / entropy_constant) ** (1.0 / gamma),
        pressure_downstream,
        tuple(float(x) for x in final[:3]),
        (bn, float(final[3]), float(final[4])),
    )
    up_speed = upstream.u[0] + direction * float(characteristic_speeds(upstream, gamma)[family])
    down_speed = downstream.u[0] + direction * float(characteristic_speeds(downstream, gamma)[family])
    monotonic = (-direction * (down_speed - up_speed)) >= -1.0e-9 * max(1.0, abs(up_speed), abs(down_speed))
    entropy_error = entropy_over_cv(downstream, gamma) - entropy_over_cv(upstream, gamma)
    sampled_speeds = []
    for pressure, vec in zip(integrated.t, integrated.y.T):
        st = State((pressure / entropy_constant) ** (1 / gamma), pressure, tuple(vec[:3]), (bn, *vec[3:]))
        sampled_speeds.append(st.u[0] + direction * float(characteristic_speeds(st, gamma)[family]))
    monotonic = monotonic and bool(np.all(-direction * np.diff(sampled_speeds) >= -1e-9 * max(1., *map(abs, sampled_speeds))))
    return CurveResult(
        downstream=downstream,
        structure="rarefaction",
        family=f"{family}_{'minus' if direction < 0 else 'plus'}",
        speed=(min(up_speed, down_speed), max(up_speed, down_speed)),
        checks={
            "entropy_change_over_cv": entropy_error,
            "isentropic": abs(entropy_error) <= 1.0e-9,
            "characteristic_speed_monotonic": monotonic,
            "ode_method": "DOP853",
            "ode_rtol": 1.0e-10,
            "ode_atol": 1.0e-12,
            "nfev": int(integrated.nfev),
            "tangent": "analytic_ideal_MHD_pressure_parameterization",
            "monotonicity_nodes": len(sampled_speeds),
        },
    )


def analytic_pressure_tangent(state: State, gamma: float, family: str, direction: int) -> np.ndarray:
    """Pressure-normalized fast/slow right eigenvector away from degeneracy.

    Ordered (u_n,u_t1,u_t2,B_t1,B_t2). Independent finite-difference
    eigensystem above is retained for validation, not used inside the ODE.
    """
    c = float(characteristic_speeds(state, gamma)[family])
    bn, bt = state.B[0], np.asarray(state.B[1:])
    denom = state.rho*c*c-bn*bn
    if abs(denom) <= 1e-12*max(1., state.rho*c*c, bn*bn):
        raise RuntimeError('ODE_EVENT_DEGENERACY')
    gp = gamma*state.p
    return np.concatenate(([direction*c/gp], -direction*bn*bt*c/(gp*denom), bt*state.rho*c*c/(gp*denom)))


def _shock_candidate(upstream: State, pressure_downstream: float, log_values: np.ndarray, direction: int) -> tuple[State, float]:
    density = math.exp(float(log_values[0]))
    mass_flux = -direction * math.exp(float(log_values[1]))
    denominator = mass_flux * mass_flux / density - upstream.B[0] ** 2
    if abs(denominator) <= 1.0e-14:
        raise FloatingPointError("JACOBIAN_SINGULAR")
    numerator = mass_flux * mass_flux / upstream.rho - upstream.B[0] ** 2
    ratio = numerator / denominator
    bt = (upstream.B[1] * ratio, upstream.B[2] * ratio)
    ut = (
        upstream.u[1] + upstream.B[0] * (bt[0] - upstream.B[1]) / mass_flux,
        upstream.u[2] + upstream.B[0] * (bt[1] - upstream.B[2]) / mass_flux,
    )
    speed = upstream.u[0] - mass_flux / upstream.rho
    un = speed + mass_flux / density
    return State(density, pressure_downstream, (un, *ut), (upstream.B[0], *bt)), speed


def solve_shock_to_pressure(
    upstream: State,
    pressure_downstream: float,
    gamma: float,
    family: str,
    direction: int,
    search_starts: bool = True,
    deadline: float | None = None,
) -> CurveResult:
    if family not in {"fast", "slow"} or direction not in {-1, 1}:
        raise ValueError("invalid shock family or direction")
    if pressure_downstream < upstream.p:
        raise ValueError("shock path requires p_downstream >= p_upstream")
    speed_scale = float(characteristic_speeds(upstream, gamma)[family])
    density_guess = upstream.rho * (pressure_downstream / upstream.p) ** (1.0 / gamma)

    def residual(values: np.ndarray) -> np.ndarray:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("SEARCH_DOMAIN_TRUNCATED")
        try:
            state, speed = _shock_candidate(upstream, pressure_downstream, values, direction)
            raw = flux(state, gamma) - flux(upstream, gamma) - speed * (
                conservative(state, gamma) - conservative(upstream, gamma)
            )
            scale = np.maximum(1.0, np.maximum(np.abs(flux(upstream, gamma)), np.abs(flux(state, gamma))))
            return np.array([raw[1] / scale[1], raw[6] / scale[6]])
        except (FloatingPointError, OverflowError, ValueError):
            return np.array([1.0e6, 1.0e6])

    lower = np.log([upstream.rho * (1.0 + 1.0e-8), upstream.rho * 1.0e-6])
    upper = np.log([upstream.rho * 1.0e6, upstream.rho * 1.0e3])
    fitted_roots: list[tuple[float, float, State, float, object]] = []
    density_factors = (1.0001, 1.2, 2.0, 3.0, 4.0, density_guess / upstream.rho) if search_starts else (density_guess / upstream.rho,)
    speed_factors = (0.5, 0.8, 1.0, 1.5, 2.0, 3.0) if search_starts else (1.0,)
    for density_factor in density_factors:
        for speed_factor in speed_factors:
            initial = np.log([
                min(upstream.rho * 9.0e5, max(upstream.rho * 1.0001, upstream.rho * density_factor)),
                max(upstream.rho * 1.0e-5, upstream.rho * speed_scale * speed_factor),
            ])
            fitted = least_squares(
                residual, initial, bounds=(lower, upper), method="trf",
                xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12, max_nfev=1000,
            )
            downstream, speed = _shock_candidate(upstream, pressure_downstream, fitted.x, direction)
            rh_error = scaled_inf_residual(upstream, downstream, speed, gamma)
            entropy_change = entropy_over_cv(downstream, gamma) - entropy_over_cv(upstream, gamma)
            fitted_roots.append((rh_error, entropy_change, downstream, speed, fitted))

    def family_ok(item: tuple[float, float, State, float, object]) -> bool:
        _, _, downstream, speed, _ = item
        up_relative = abs(upstream.u[0] - speed)
        down_relative = abs(downstream.u[0] - speed)
        up_speeds = characteristic_speeds(upstream, gamma)
        down_speeds = characteristic_speeds(downstream, gamma)
        if family == "fast":
            return up_relative >= float(up_speeds["fast"]) * (1.0 - 1.0e-8) and down_relative <= float(down_speeds["fast"]) * (1.0 + 1.0e-8)
        return (
            up_relative >= float(up_speeds["slow"]) * (1.0 - 1.0e-8)
            and up_relative <= float(up_speeds["alfven_n"]) * (1.0 + 1.0e-8)
            and down_relative <= float(down_speeds["slow"]) * (1.0 + 1.0e-8)
        )

    admissible = [item for item in fitted_roots if item[0] <= 1.0e-10 and item[1] >= -1.0e-10 and family_ok(item)]
    if not admissible:
        raise RuntimeError("ROOT_NOT_CONVERGED")
    rh_error, entropy_change, downstream, speed, fitted = min(admissible, key=lambda item: (item[0], abs(item[2].rho - density_guess)))
    unique_roots: list[tuple[float, float]] = []
    for item in fitted_roots:
        key = (item[2].rho, abs(upstream.rho * (upstream.u[0] - item[3])))
        if item[0] <= 1.0e-8 and all(np.linalg.norm(np.array(key) - np.array(old)) > 1.0e-8 * max(1.0, *map(abs, key)) for old in unique_roots):
            unique_roots.append(key)
    return CurveResult(
        downstream=downstream,
        structure="shock",
        family=f"{family}_{'minus' if direction < 0 else 'plus'}",
        speed=speed,
        checks={
            "rh_scaled_inf": rh_error,
            "entropy_change_over_cv": entropy_change,
            "entropy_admissible": entropy_change >= -1.0e-10,
            "root_success": bool(fitted.success),
            "root_cost": float(fitted.cost),
            "nfev": int(fitted.nfev),
            "starts": len(density_factors) * len(speed_factors),
            "converged_root_clusters": len(unique_roots),
        },
    )


def solve_magnetosonic_to_pressure(
    upstream: State,
    pressure_downstream: float,
    gamma: float,
    family: str,
    direction: int,
    search_starts: bool = True,
    deadline: float | None = None,
) -> CurveResult:
    if math.isclose(pressure_downstream, upstream.p, rel_tol=5.0e-9, abs_tol=5.0e-9):
        return CurveResult(upstream, "zero_strength", f"{family}_{'minus' if direction < 0 else 'plus'}", upstream.u[0], {"exact": True})
    if pressure_downstream > upstream.p:
        return solve_shock_to_pressure(upstream, pressure_downstream, gamma, family, direction, search_starts, deadline)
    return solve_rarefaction_to_pressure(upstream, pressure_downstream, gamma, family, direction, deadline)


def apply_rotation(state: State, angle: float, direction: int, gamma: float) -> CurveResult:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    bt1 = state.B[1] * cos_angle - state.B[2] * sin_angle
    bt2 = state.B[1] * sin_angle + state.B[2] * cos_angle
    # The RH sign is fixed by the selected Alfvén characteristic.
    rotation_sign = 1.0 if (direction < 0) == (state.B[0] >= 0.0) else -1.0
    ut1 = state.u[1] + rotation_sign * (bt1 - state.B[1]) / math.sqrt(state.rho)
    ut2 = state.u[2] + rotation_sign * (bt2 - state.B[2]) / math.sqrt(state.rho)
    downstream = State(state.rho, state.p, (state.u[0], ut1, ut2), (state.B[0], bt1, bt2))
    speed = state.u[0] + direction * abs(state.B[0]) / math.sqrt(state.rho)
    return CurveResult(
        downstream, "rotation", f"alfven_{'minus' if direction < 0 else 'plus'}", speed,
        {"rh_scaled_inf": scaled_inf_residual(state, downstream, speed, gamma), "rotation_angle": angle},
    )
