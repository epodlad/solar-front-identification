from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Iterable

from .models import State


REGULAR_POLICY = "REGULAR_EVOLUTIONARY_1.0"
NONREGULAR_POLICY = "ENUMERATE_NONREGULAR_1.0"


@dataclass(frozen=True)
class WaveSlot:
    family: str
    structure: str
    direction: int
    attachment: str | None = None


@dataclass(frozen=True)
class FanTopology:
    topology_id: str
    policy: str
    waves: tuple[WaveSlot, ...]
    continuous_parameters: tuple[str, ...] = ()
    regular: bool = True
    provenance: str = "RMO-R6A-SPEC-1.0.0"

    def serializable(self) -> dict[str, object]:
        value = asdict(self)
        value["waves"] = [asdict(wave) for wave in self.waves]
        value["continuous_parameters"] = list(self.continuous_parameters)
        return value


@dataclass(frozen=True)
class GeometryClass:
    bn_zero: bool
    left_bt_zero: bool
    right_bt_zero: bool
    coplanar: bool
    transverse_reversal: bool

    def serializable(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyPlan:
    geometry: GeometryClass
    policy: str
    candidates: tuple[FanTopology, ...]
    exhaustive_at_topology_level: bool
    continuous_solution_family_possible: bool
    unresolved_requirements: tuple[str, ...]

    def serializable(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.serializable(),
            "policy": self.policy,
            "candidate_count": len(self.candidates),
            "candidates": [item.serializable() for item in self.candidates],
            "exhaustive_at_topology_level": self.exhaustive_at_topology_level,
            "continuous_solution_family_possible": self.continuous_solution_family_possible,
            "unresolved_requirements": list(self.unresolved_requirements),
        }


def classify_geometry(left: State, right: State, tol: float = 1.0e-12) -> GeometryClass:
    left_bt = left.B[1:]
    right_bt = right.B[1:]
    left_norm = math.hypot(*left_bt)
    right_norm = math.hypot(*right_bt)
    cross = left_bt[0] * right_bt[1] - left_bt[1] * right_bt[0]
    dot = left_bt[0] * right_bt[0] + left_bt[1] * right_bt[1]
    scale = max(1.0, left_norm * right_norm)
    coplanar = abs(cross) <= tol * scale
    return GeometryClass(
        bn_zero=max(abs(left.B[0]), abs(right.B[0])) <= tol,
        left_bt_zero=left_norm <= tol,
        right_bt_zero=right_norm <= tol,
        coplanar=coplanar,
        transverse_reversal=coplanar and dot < -(tol * scale),
    )


def _regular_topologies(policy: str) -> Iterable[FanTopology]:
    magnetosonic = ("shock", "rarefaction", "zero_strength")
    rotations = ("rotation", "zero_strength")
    for lf, la, ls, rs, ra, rf in product(
        magnetosonic, rotations, magnetosonic, magnetosonic, rotations, magnetosonic
    ):
        values = (lf, la, ls, rs, ra, rf)
        topology_id = "REG-" + "-".join(item[:2].upper() for item in values)
        yield FanTopology(
            topology_id=topology_id,
            policy=policy,
            waves=(
                WaveSlot("fast_minus", lf, -1),
                WaveSlot("alfven_minus", la, -1),
                WaveSlot("slow_minus", ls, -1),
                WaveSlot("entropy", "contact", 0),
                WaveSlot("slow_plus", rs, 1),
                WaveSlot("alfven_plus", ra, 1),
                WaveSlot("fast_plus", rf, 1),
            ),
        )


def _nonregular_catalogue(policy: str) -> Iterable[FanTopology]:
    """Enumerate the non-regular topology catalogue frozen for R6c search.

    This catalogue is deterministic but is not itself a proof that every
    continuous strength branch has been numerically traced.
    """

    intermediate_types = ("1_to_3", "1_to_4", "2_to_3", "2_to_4")
    for direction, suffix in ((-1, "L"), (1, "R")):
        for kind in intermediate_types:
            yield FanTopology(
                topology_id=f"NR-{suffix}-IS-{kind}",
                policy=policy,
                waves=(WaveSlot("intermediate", f"intermediate_{kind}", direction),),
                continuous_parameters=(f"{suffix.lower()}_intermediate_strength",),
                regular=False,
                provenance="Takahashi-Yamada-2014-sections-4-6",
            )
        for kind in ("switch_on_shock", "switch_off_shock", "switch_off_rarefaction"):
            yield FanTopology(
                topology_id=f"NR-{suffix}-{kind.upper()}",
                policy=policy,
                waves=(WaveSlot("intermediate", kind, direction),),
                regular=False,
                provenance="Takahashi-Yamada-2014-sections-2.3-4",
            )
    yield FanTopology(
        topology_id="NR-L-COMPOUND-2_TO_3_4-SR",
        policy=policy,
        waves=(
            WaveSlot("fast_minus", "rarefaction", -1),
            WaveSlot("intermediate", "intermediate_2_to_3_4", -1, attachment="slow_rarefaction"),
            WaveSlot("slow_minus", "rarefaction", -1, attachment="intermediate_2_to_3_4"),
            WaveSlot("entropy", "contact", 0),
            WaveSlot("slow_plus", "shock", 1),
            WaveSlot("fast_plus", "rarefaction", 1),
        ),
        continuous_parameters=("left_intermediate_strength",),
        regular=False,
        provenance="Takahashi-Yamada-2013-Brio-Wu-section-5.1",
    )
    yield FanTopology(
        topology_id="NR-L-COMPOUND-1_2_TO_3-FR",
        policy=policy,
        waves=(
            WaveSlot("fast_minus", "rarefaction", -1, attachment="intermediate_1_2_to_3"),
            WaveSlot("intermediate", "intermediate_1_2_to_3", -1, attachment="fast_rarefaction"),
            WaveSlot("entropy", "contact", 0),
            WaveSlot("slow_plus", "shock", 1),
            WaveSlot("fast_plus", "shock_or_rarefaction", 1),
        ),
        continuous_parameters=("left_intermediate_strength",),
        regular=False,
        provenance="Takahashi-Yamada-2014-section-6",
    )


def enumerate_topologies(left: State, right: State, policy: str) -> TopologyPlan:
    geometry = classify_geometry(left, right)
    regular = tuple(_regular_topologies(policy))
    if policy == REGULAR_POLICY:
        return TopologyPlan(
            geometry=geometry,
            policy=policy,
            candidates=regular,
            exhaustive_at_topology_level=not geometry.bn_zero and not geometry.left_bt_zero and not geometry.right_bt_zero,
            continuous_solution_family_possible=False,
            unresolved_requirements=("wave_curve_matching", "root_set_stability", "cross_code_validation"),
        )
    if policy != NONREGULAR_POLICY:
        raise ValueError(f"unknown policy: {policy}")
    nonregular = tuple(_nonregular_catalogue(policy))
    continuum = geometry.coplanar and geometry.transverse_reversal
    return TopologyPlan(
        geometry=geometry,
        policy=policy,
        candidates=regular + nonregular,
        # The published method warns that isolated roots may evade continuation;
        # topology cataloguing therefore cannot certify numerical completeness.
        exhaustive_at_topology_level=False,
        continuous_solution_family_possible=continuum,
        unresolved_requirements=(
            "nonregular_wave_curve_evaluators",
            "continuous_strength_continuation",
            "coupled_contact_matching",
            "root_set_stability",
            "cross_code_validation",
        ),
    )
