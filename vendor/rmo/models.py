from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class State:
    rho: float
    p: float
    u: tuple[float, float, float]
    B: tuple[float, float, float]
    id: str = ""
    role: str = "intermediate"
    frame: str = "RIEMANN_COMPUTATIONAL"
    units: str = "normalized_mu0_1"

    @classmethod
    def from_mapping(cls, value: dict[str, Any], **overrides: Any) -> "State":
        fields = dict(value)
        fields.update(overrides)
        fields["u"] = tuple(float(x) for x in fields["u"])
        fields["B"] = tuple(float(x) for x in fields["B"])
        return cls(**fields)

    @property
    def uvec(self) -> np.ndarray:
        return np.asarray(self.u, dtype=float)

    @property
    def bvec(self) -> np.ndarray:
        return np.asarray(self.B, dtype=float)

    def serializable(self) -> dict[str, Any]:
        out = asdict(self)
        out["u"] = list(self.u)
        out["B"] = list(self.B)
        return out


@dataclass
class Wave:
    order: int
    structure: str
    family: str
    speed: float | tuple[float, float]
    left_state: State
    right_state: State
    checks: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "structure": self.structure,
            "family": self.family,
            "speed": list(self.speed) if isinstance(self.speed, tuple) else self.speed,
            "left_state_ref": self.left_state.id,
            "right_state_ref": self.right_state.id,
            "checks": self.checks,
        }


@dataclass
class Solution:
    id: str
    policy: str
    waves: list[Wave]
    intermediate_states: list[State]
    domain_codes: list[str] = field(default_factory=lambda: ["OK"])
    root_provenance: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "policy": self.policy,
            "waves": [w.serializable() for w in self.waves],
            "intermediate_states": [s.serializable() for s in self.intermediate_states],
            "domain_codes": self.domain_codes,
            "root_provenance": self.root_provenance,
        }


@dataclass
class SolveResult:
    specification_id: str
    policy: str
    solutions: list[Solution] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    domain_codes: list[str] = field(default_factory=list)
    complete: bool = False

    def serializable(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "policy": self.policy,
            "solutions": [s.serializable() for s in self.solutions],
            "rejected_candidates": self.rejected_candidates,
            "domain_codes": self.domain_codes,
            "complete": self.complete,
        }
