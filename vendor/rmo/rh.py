from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .models import State
from .physics import characteristic_speeds, entropy_over_cv, scaled_inf_residual


@dataclass
class ShockBranch:
    compression: float
    downstream: State
    family: str
    evolutionary: bool
    entropy_change_over_cv: float
    scaled_residual: float


def downstream_from_compression(up: State, compression: float, gamma: float) -> State:
    del gamma
    if compression <= 0.0:
        raise ValueError("compression must be positive")
    rho2 = compression * up.rho
    un2 = up.u[0] / compression
    mass_flux = up.rho * up.u[0]
    bt2: list[float] = []
    ut2: list[float] = []
    for bt, ut in zip(up.B[1:], up.u[1:]):
        electric = up.u[0] * bt - up.B[0] * ut
        tangential_momentum = mass_flux * ut - up.B[0] * bt
        matrix = np.array([[un2, -up.B[0]], [-up.B[0], mass_flux]], dtype=float)
        rhs = np.array([electric, tangential_momentum], dtype=float)
        solved_bt, solved_ut = np.linalg.solve(matrix, rhs)
        bt2.append(float(solved_bt))
        ut2.append(float(solved_ut))
    p2 = (
        up.p
        + up.rho * up.u[0] ** 2
        - rho2 * un2**2
        + 0.5 * (sum(x * x for x in up.B[1:]) - sum(x * x for x in bt2))
    )
    return State(rho2, p2, (un2, *ut2), (up.B[0], *bt2), role="downstream")


def energy_flux(state: State, gamma: float) -> float:
    from .physics import total_energy, total_pressure

    return (
        (total_energy(state, gamma) + total_pressure(state)) * state.u[0]
        - state.B[0] * float(np.dot(state.uvec, state.bvec))
    )


def energy_residual(up: State, compression: float, gamma: float) -> float:
    down = downstream_from_compression(up, compression, gamma)
    return energy_flux(down, gamma) - energy_flux(up, gamma)


def classify_shock(up: State, down: State, gamma: float, tol: float = 1.0e-9) -> tuple[str, bool]:
    cu = characteristic_speeds(up, gamma)
    cd = characteristic_speeds(down, gamma)
    u1, u2 = abs(up.u[0]), abs(down.u[0])
    fast = u1 > float(cu["fast"]) * (1.0 - tol) and u2 < float(cd["fast"]) * (1.0 + tol) and u2 >= float(cd["alfven_n"]) * (1.0 - tol)
    slow = u1 > float(cu["slow"]) * (1.0 - tol) and u1 <= float(cu["alfven_n"]) * (1.0 + tol) and u2 < float(cd["slow"]) * (1.0 + tol)
    if fast:
        return "fast", True
    if slow:
        return "slow", True
    return "intermediate_or_nonevolutionary", False


def solve_shock_branches(
    upstream: State,
    gamma: float,
    r_min: float = 1.000001,
    r_max: float | None = None,
    samples: int = 4097,
) -> list[ShockBranch]:
    branches, _ = solve_shock_branches_with_provenance(upstream, gamma, r_min, r_max, samples)
    return branches


def solve_shock_branches_with_provenance(
    upstream: State,
    gamma: float,
    r_min: float = 1.000001,
    r_max: float | None = None,
    samples: int = 4097,
) -> tuple[list[ShockBranch], dict[str, object]]:
    if r_max is None:
        r_max = 1.25 * (gamma + 1.0) / (gamma - 1.0)
    grid = np.linspace(r_min, r_max, samples)
    values: list[float] = []
    invalid_nodes = 0
    for compression in grid:
        try:
            down = downstream_from_compression(upstream, float(compression), gamma)
            value = energy_residual(upstream, float(compression), gamma) if down.p > 0.0 else math.nan
            invalid_nodes += int(not math.isfinite(value))
            values.append(value)
        except (np.linalg.LinAlgError, ValueError):
            invalid_nodes += 1
            values.append(math.nan)
    roots: list[float] = []
    brackets: list[list[float]] = []
    rejected_roots: list[dict[str, object]] = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not (math.isfinite(fa) and math.isfinite(fb)) or fa * fb > 0.0:
            continue
        brackets.append([float(a), float(b)])
        try:
            root = float(a if fa == 0.0 else brentq(lambda x: energy_residual(upstream, x, gamma), a, b, xtol=1.0e-13, rtol=1.0e-14))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rejected_roots.append({"bracket": [float(a), float(b)], "reason": type(exc).__name__})
            continue
        if root <= r_min:
            rejected_roots.append({"root": root, "reason": "zero_strength_or_domain_boundary"})
        elif any(abs(root - old) <= 1.0e-8 for old in roots):
            rejected_roots.append({"root": root, "reason": "root_cluster_duplicate"})
        else:
            roots.append(root)
    branches: list[ShockBranch] = []
    for root in roots:
        down = downstream_from_compression(upstream, root, gamma)
        family, evolutionary = classify_shock(upstream, down, gamma)
        delta_s = entropy_over_cv(down, gamma) - entropy_over_cv(upstream, gamma)
        branches.append(
            ShockBranch(
                compression=root,
                downstream=down,
                family=family,
                evolutionary=evolutionary and delta_s >= -1.0e-10,
                entropy_change_over_cv=delta_s,
                scaled_residual=scaled_inf_residual(upstream, down, 0.0, gamma),
            )
        )
    provenance: dict[str, object] = {
        "algorithm": "compression_scan_plus_brentq",
        "samples": samples,
        "compression_domain": [float(r_min), float(r_max)],
        "invalid_nodes": invalid_nodes,
        "sign_change_brackets": brackets,
        "accepted_roots": roots,
        "rejected_roots": rejected_roots,
        "root_cluster_tolerance": 1.0e-8,
        "brentq_xtol": 1.0e-13,
        "brentq_rtol": 1.0e-14,
    }
    return branches, provenance
