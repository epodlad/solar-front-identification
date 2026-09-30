from __future__ import annotations

from dataclasses import replace

from .discontinuities import recognize_constant, recognize_contact, recognize_rotation, recognize_tangential
from .hydro import exact_hydro_riemann, is_hydrodynamic, vacuum_condition
from .models import SolveResult, State
from .physics import SPECIFICATION_ID, validate_pair
from .topology import enumerate_topologies


POLICIES = {"REGULAR_EVOLUTIONARY_1.0", "ENUMERATE_NONREGULAR_1.0"}


class RiemannSolver:
    """Fail-closed R6b prototype.

    Exact for the hydro subset and analytic MHD discontinuity fixtures. General
    seven-wave ideal-MHD branch matching is intentionally reported incomplete
    until a validated all-branch implementation exists.
    """

    def solve(self, left: State, right: State, gamma: float, policy: str) -> SolveResult:
        if policy not in POLICIES:
            return SolveResult(SPECIFICATION_ID, policy, domain_codes=["INVALID_STATE"], complete=False)
        left = replace(left, id=left.id or "X_L0", role="initial_left")
        right = replace(right, id=right.id or "X_R0", role="initial_right")
        codes = validate_pair(left, right, gamma)
        if codes != ["OK"]:
            return SolveResult(SPECIFICATION_ID, policy, domain_codes=codes, complete=False)
        if vacuum_condition(left, right, gamma):
            return SolveResult(SPECIFICATION_ID, policy, domain_codes=["VACUUM_FORMED"], complete=True)
        for recognizer in (recognize_constant,):
            solution = recognizer(left, right, policy)
            if solution is not None:
                return SolveResult(SPECIFICATION_ID, policy, [solution], domain_codes=["OK"], complete=True)
        for recognizer in (recognize_contact, recognize_rotation, recognize_tangential):
            solution = recognizer(left, right, gamma, policy)
            if solution is not None:
                return SolveResult(SPECIFICATION_ID, policy, [solution], domain_codes=["OK"], complete=True)
        if is_hydrodynamic(left, right):
            try:
                solution = exact_hydro_riemann(left, right, gamma, policy)
            except ValueError as exc:
                return SolveResult(SPECIFICATION_ID, policy, domain_codes=[str(exc)], complete=True)
            except RuntimeError as exc:
                return SolveResult(SPECIFICATION_ID, policy, domain_codes=[str(exc)], complete=False)
            return SolveResult(SPECIFICATION_ID, policy, [solution], domain_codes=["OK"], complete=True)

        plan = enumerate_topologies(left, right, policy)
        codes = ["BRANCH_ENUMERATION_INCOMPLETE"]
        if policy == "ENUMERATE_NONREGULAR_1.0":
            codes.insert(0, "POLICY_DEPENDENT")
        return SolveResult(
            SPECIFICATION_ID,
            policy,
            rejected_candidates=[
                {
                    "candidate": "general_ideal_mhd_complete_fan",
                    "failed_predicate": "declared_topologies_not_all_matched",
                    "preserved_initial_states": [left.serializable(), right.serializable()],
                    "topology_plan": plan.serializable(),
                }
            ],
            domain_codes=codes,
            complete=False,
        )

    def trace_coplanar_nonregular(self, left: State, right: State, gamma: float,
                                 ratios, initial_guess, *, seconds: float=120.) -> SolveResult:
        """Explicit targeted continuation; never an all-topology solve.

        ratios and initial_guess are declared by caller, with no hidden
        Brio--Wu initial-state replacement or gamma override.
        """
        from .continuation import Budget,correct_at_ratio
        codes=validate_pair(left,right,gamma)
        if codes!=['OK']:
            return SolveResult(SPECIFICATION_ID,'ENUMERATE_NONREGULAR_1.0',domain_codes=codes,complete=False)
        budget=Budget(seconds)
        solutions=[]; records=[]; guess=initial_guess
        try:
            for ratio in ratios:
                sol,record=correct_at_ratio(left,right,gamma,float(ratio),guess,budget)
                records.append(record)
                if sol is None: break
                solutions.append(sol); guess=sol.root_provenance['parameters']
        except (TimeoutError,ValueError) as exc:
            records.append({'reason':str(exc),'starts':budget.starts})
        codes=['POLICY_DEPENDENT','BRANCH_ENUMERATION_INCOMPLETE']
        if any(r.get('reason')=='SEARCH_DOMAIN_TRUNCATED' for r in records): codes.append('SEARCH_DOMAIN_TRUNCATED')
        for sol in solutions:
            sol.root_provenance['continuation_attempt_records']=records
        # All attempts, including accepted correctors, remain provenance;
        # only failed attempts and the global limitation are rejection records.
        rejected=[r for r in records if r.get('reason')!='accepted']
        rejected.append({'candidate':'all_other_topologies','failed_predicate':'not_enumerated_by_targeted_trace'})
        return SolveResult(SPECIFICATION_ID,'ENUMERATE_NONREGULAR_1.0',solutions,rejected,codes,False)
