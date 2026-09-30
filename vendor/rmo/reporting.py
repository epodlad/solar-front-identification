from __future__ import annotations

import hashlib
import json
import platform
from typing import Any

import numpy as np
import scipy

from .models import SolveResult, State


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_result_record(
    left: State,
    right: State,
    gamma: float,
    result: SolveResult,
    *,
    seed: int = 20260905,
) -> dict[str, Any]:
    """Wrap a solver result in the frozen RMO result contract.

    Synthetic R6b runs contain no observations. Assessment remains explicit:
    incomplete enumeration cannot yield a unique family classification.
    """

    complete = result.complete and not result.domain_codes or result.complete and result.domain_codes == ["OK"]
    status = "complete" if complete else "unresolved"
    config = {"gamma": gamma, "policy": result.policy, "seed": seed}
    return {
        "specification": {"id": result.specification_id, "schema_version": "rmo-result-1.1.0"},
        "physics": {
            "model": "ideal_mhd_1d",
            "gamma": gamma,
            "mu0_normalized": 1,
            "admissibility_policy": result.policy,
        },
        "normalization": {"rho0": 1.0, "v0": 1.0, "B0": 1.0, "p0": 1.0, "source_units": "synthetic_normalized"},
        "geometry": {
            "source_frame": "RIEMANN_COMPUTATIONAL",
            "solver_frame": "RIEMANN_COMPUTATIONAL",
            "normal": [1.0, 0.0, 0.0],
            "t1": [0.0, 1.0, 0.0],
            "t2": [0.0, 0.0, 1.0],
            "galilean_offset": [0.0, 0.0, 0.0],
            "handedness": "right",
        },
        "initial_states": {"left": left.serializable(), "right": right.serializable()},
        "observations": [],
        "uncertainty": {
            "representation": "posterior_samples",
            "correlations_preserved": True,
            "censoring_preserved": True,
            "sample_count": 1,
            "seed": seed,
        },
        "policies": [result.policy],
        "solutions": [solution.serializable() for solution in result.solutions],
        "assessment": {
            "feature_identity": {"status": "synthetic_fixture"},
            "model_domain": {"status": status, "codes": result.domain_codes},
            "mathematical_structure": {"status": status},
            "mhd_family": {"status": "unresolved" if not complete else "compatible"},
            "sequence_identifiability": {"status": "unresolved" if not complete else "enumerated"},
            "evidence_strength": {"status": "synthetic_validation_only"},
            "survivors": [solution.id for solution in result.solutions],
            "vetoes": list(result.domain_codes),
            "rejected_candidates": result.rejected_candidates,
        },
        "provenance": {
            "source_hashes": [_hash_json(left.serializable()), _hash_json(right.serializable())],
            "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "configuration_hash": _hash_json(config),
            "random_seed": seed,
        },
    }
