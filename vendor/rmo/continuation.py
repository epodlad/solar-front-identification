"""Coplanar non-regular fan continuation, not an exhaustive MHD solver.

Topology: left fast wave, field-reversing intermediate/rotation, left slow
wave, contact, right slow wave, right fast wave. Initial Bz=uz=0 is required.
Unknowns are four wave strengths; the intermediate Bt ratio is continued.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import least_squares
from .models import State, Solution, Wave
from .nonregular import hugoniot_at_ratio, shock_checks
from .physics import scaled_inf_residual, validate_pair, characteristic_speeds
from .wave_curves import CurveResult, solve_magnetosonic_to_pressure, solve_rarefaction_to_pressure


@dataclass
class Budget:
    seconds: float = 1800.
    max_starts: int = 10000
    starts: int = 0
    def __post_init__(self):
        if not (0 < self.seconds <= 1800.) or not (0 < self.max_starts <= 10000):
            raise ValueError('RESOURCE_CEILING_EXCEEDED')
        self.began = time.monotonic()
        self.deadline = self.began + self.seconds
    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('SEARCH_DOMAIN_TRUNCATED')
    def start(self):
        self.check()
        if self.starts >= self.max_starts:
            raise TimeoutError('SEARCH_DOMAIN_TRUNCATED')
        self.starts += 1


def slow_curve(up: State, strength: float, gamma: float, direction: int, budget: Budget) -> CurveResult:
    budget.check()
    if strength <= 0:
        return solve_magnetosonic_to_pressure(up, up.p*math.exp(strength), gamma, 'slow', direction, deadline=budget.deadline)
    candidates = hugoniot_at_ratio(up, math.exp(-strength), gamma, direction).accepted
    candidates = [x for x in candidates if x.checks.get('transition') == '3_to_4']
    if len(candidates) != 1:
        raise ValueError('SLOW_BRANCH_UNRESOLVED')
    return candidates[0]


def build_fan_curves(left: State, right: State, gamma: float, ratio: float, x: np.ndarray, budget: Budget, root_index: int = 0):
    budget.check()
    lf = solve_magnetosonic_to_pressure(left, left.p*math.exp(float(x[0])), gamma, 'fast', -1, search_starts=False, deadline=budget.deadline)
    options = hugoniot_at_ratio(lf.downstream, ratio, gamma, -1).accepted
    if ratio == -1:
        options = [v for v in options if v.structure == 'rotation']
    else:
        options = [v for v in options if v.structure == 'intermediate_shock']
    if root_index >= len(options):
        raise ValueError('INTERMEDIATE_BRANCH_UNRESOLVED')
    intermediate = options[root_index]
    ls = slow_curve(intermediate.downstream, float(x[1]), gamma, -1, budget)
    rf = solve_magnetosonic_to_pressure(right, right.p*math.exp(float(x[2])), gamma, 'fast', 1, search_starts=False, deadline=budget.deadline)
    rs = slow_curve(rf.downstream, float(x[3]), gamma, 1, budget)
    return [lf, intermediate, ls], [rf, rs]


def matching_residual(left, right, gamma, ratio, x, budget, root_index=0):
    budget.check()
    try:
        l,r = build_fan_curves(left,right,gamma,ratio,x,budget,root_index)
        a,b = l[-1].downstream,r[-1].downstream
        va,vb = np.array([a.p,a.u[0],a.u[1],a.B[1]]), np.array([b.p,b.u[0],b.u[1],b.B[1]])
        return (va-vb)/np.maximum(1.,np.maximum(abs(va),abs(vb)))
    except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
        budget.check()
        return np.full(4,1e3)


def reconstruct(left,right,gamma,ratio,x,budget,root_index=0):
    lc,rc = build_fan_curves(left,right,gamma,ratio,x,budget,root_index)
    l_states = [replace(left,id='X_L0',role='initial_left')]
    r_states = [replace(right,id='X_R0',role='initial_right')]
    for i,c in enumerate(lc): l_states.append(replace(c.downstream,id=f'X_L{i+1}'))
    for i,c in enumerate(rc): r_states.append(replace(c.downstream,id=f'X_R{i+1}'))
    waves = []
    for i,c in enumerate(lc):
        fam = 'fast_minus' if i==0 else 'alfven_minus' if c.structure=='rotation' else 'intermediate' if i==1 else 'slow_minus'
        # Preserve frozen axes: mathematical structure=shock; MHD family=intermediate.
        structure='shock' if c.structure in ('intermediate_shock','switch_off_shock','switch_on_shock') else c.structure
        checks=dict(c.checks)
        if structure!=c.structure: checks['shock_subtype']=c.structure
        waves.append(Wave(len(waves),structure,fam,c.speed,l_states[i],l_states[i+1],checks))
    s = l_states[-1].u[0]
    waves.append(Wave(3,'contact','entropy',s,l_states[-1],r_states[-1],{'rh_scaled_inf':scaled_inf_residual(l_states[-1],r_states[-1],s,gamma)}))
    for i in (1,0):
        c=rc[i]
        waves.append(Wave(len(waves),c.structure,'slow_plus' if i==1 else 'fast_plus',c.speed,r_states[i+1],r_states[i],dict(c.checks)))
    policy = 'REGULAR_EVOLUTIONARY_1.0' if ratio == -1 else 'ENUMERATE_NONREGULAR_1.0'
    sol = Solution('coplanar_'+format(ratio,'.12g'),policy,waves,l_states[1:]+r_states[1:], ['OK'] if ratio==-1 else ['POLICY_DEPENDENT'],{
        'method':'four_wave_strength_contact_matching_at_fixed_intermediate_Bt_ratio',
        'intermediate_Bt_ratio':ratio,'parameters':x.tolist(),'algebraic_branch_index':root_index,
        'contact_scaled_inf':float(max(abs(matching_residual(left,right,gamma,ratio,x,budget,root_index)))),
        'full_topology_enumeration_complete':False,
    })
    return sol


def validate_fan(sol: Solution, gamma: float) -> dict:
    reasons=[]; bounds=[]; rhmax=0.
    for w in sol.waves:
        # A zero-strength family has no physical interval to order.
        if w.structure=='zero_strength': continue
        bounds.append(tuple(w.speed) if isinstance(w.speed,tuple) else (w.speed,w.speed))
        if w.structure=='rarefaction':
            if not w.checks.get('isentropic') or not w.checks.get('characteristic_speed_monotonic'):
                reasons.append('rarefaction_integral_or_order')
        else:
            err=scaled_inf_residual(w.left_state,w.right_state,float(w.speed),gamma)
            rhmax=max(rhmax,err)
            if err>1e-10: reasons.append('RH:'+str(w.order))
            if w.structure in ('shock','intermediate_shock','switch_off_shock'):
                direction=-1 if w.order<3 else 1
                up,down=(w.left_state,w.right_state) if direction<0 else (w.right_state,w.left_state)
                checks=shock_checks(up,down,float(w.speed),gamma,direction)
                w.checks.update(checks)
                if not checks['entropy_admissible']: reasons.append('entropy:'+str(w.order))
                if w.family.startswith('fast') and checks['transition']!='1_to_2': reasons.append('fast_region')
                if w.family.startswith('slow') and checks['transition']!='3_to_4': reasons.append('slow_region')
        if any(s.rho<=1e-12 or s.p<=1e-12 for s in (w.left_state,w.right_state)): reasons.append('INVALID_STATE')
    scale=max(1.,max(abs(v) for b in bounds for v in b))
    gaps=[b[0]-a[1] for a,b in zip(bounds,bounds[1:])]
    if min(gaps,default=0)<-1e-9*scale: reasons.append('wave_speed_order')
    if sol.root_provenance['contact_scaled_inf']>1e-10: reasons.append('contact_matching')
    return {'locally_valid':not reasons,'reasons':reasons,'max_RH':rhmax,'min_wave_gap':min(gaps,default=0.),'stability':'NOT_ASSESSED'}


def correct_at_ratio(left,right,gamma,ratio,guess,budget,root_index=0):
    if validate_pair(left,right,gamma)!=['OK'] or max(abs(left.B[2]),abs(right.B[2]),abs(left.u[2]),abs(right.u[2]))>1e-12:
        raise ValueError('INVALID_COPLANAR_INPUT')
    budget.start()
    # Local declared box; not a claim to span the frozen global search box.
    lo=np.array([-8.,-8.,-8.,-8.]); hi=np.array([4.,8.,4.,8.])
    guess=np.asarray(guess,dtype=float)
    if np.any(guess<=lo) or np.any(guess>=hi):
        return None,{'reason':'SEARCH_DOMAIN_TRUNCATED','ratio':ratio,'guess':guess.tolist()}
    fit=least_squares(lambda x:matching_residual(left,right,gamma,ratio,x,budget,root_index),guess,
                      bounds=(lo,hi),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
    res=float(max(abs(matching_residual(left,right,gamma,ratio,fit.x,budget,root_index))))
    record={'ratio':ratio,'start':budget.starts,'parameters':fit.x.tolist(),'residual':res,'nfev':fit.nfev,'optimizer_success':bool(fit.success)}
    if not fit.success or res>1e-10:
        return None,{**record,'reason':'ROOT_NOT_CONVERGED'}
    if np.any(abs(fit.x-lo)<1e-8) or np.any(abs(fit.x-hi)<1e-8):
        return None,{**record,'reason':'SEARCH_DOMAIN_TRUNCATED'}
    sol=reconstruct(left,right,gamma,ratio,fit.x,budget,root_index)
    validation=validate_fan(sol,gamma)
    sol.root_provenance['validation']=validation
    return (sol if validation['locally_valid'] else None),{**record,'validation':validation,'reason':'accepted' if validation['locally_valid'] else 'candidate_validation'}


def attachment_residual(left,right,gamma,z,budget,root_index=0):
    """Contact matching plus s_IS = u_post - c_s,post (left compound)."""
    budget.check()
    ratio,x=float(z[4]),z[:4]
    residual=matching_residual(left,right,gamma,ratio,x,budget,root_index)
    try:
        l,_=build_fan_curves(left,right,gamma,ratio,x,budget,root_index)
        d=l[1].downstream
        cs=float(characteristic_speeds(d,gamma)['slow'])
        gap=d.u[0]-cs-float(l[1].speed)
        return np.append(residual,gap/max(1.,cs,abs(float(l[1].speed))))
    except (ValueError,RuntimeError,FloatingPointError):
        budget.check()
        return np.full(5,1e3)


def solve_compound(left,right,gamma,guess,ratio_guess,budget,root_index=0):
    budget.start()
    fit=least_squares(lambda z:attachment_residual(left,right,gamma,z,budget,root_index),
                      np.append(guess,ratio_guess),bounds=([-8,-8,-8,-8,-.999999],[4,8,4,8,-.001]),
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    residual=float(max(abs(attachment_residual(left,right,gamma,fit.x,budget,root_index))))
    record={'parameters':fit.x.tolist(),'residual':residual,'optimizer_success':bool(fit.success),'nfev':fit.nfev}
    if not fit.success or residual>1e-10:
        return None,{**record,'reason':'ROOT_NOT_CONVERGED'}
    sol=reconstruct(left,right,gamma,float(fit.x[4]),fit.x[:4],budget,root_index)
    validation=validate_fan(sol,gamma)
    if sol.waves[2].structure!='rarefaction' or sol.waves[1].checks.get('transition')!='2_to_3,4':
        validation['locally_valid']=False
        validation['reasons'].append('compound_type_or_attachment')
    sol.root_provenance.update({'compound_attachment_residual':residual,'validation':validation})
    sol.waves[1].checks.update({'attached_to_wave_order':2,'attachment_condition':'s_IS=u_down-c_s_down','attachment_residual':residual})
    sol.waves[2].checks.update({'attached_to_wave_order':1})
    return (sol if validation['locally_valid'] else None),{**record,'validation':validation,'reason':'accepted' if validation['locally_valid'] else 'candidate_validation'}


def finite_jacobian(fun,z,step=1e-5):
    z=np.asarray(z,dtype=float)
    cols=[]
    for i in range(len(z)):
        h=step*max(1.,abs(z[i])); delta=np.zeros_like(z); delta[i]=h
        cols.append((fun(z+delta)-fun(z-delta))/(2*h))
    return np.column_stack(cols)


def pseudo_arclength_step(fun,z,ds,previous_tangent=None,budget=None):
    """Generic predictor/corrector; can pass a fold of the chosen parameter.

    The returned Jacobian spectrum diagnoses local rank only. No claim about
    disconnected branches or global uniqueness follows from a successful step.
    """
    if budget: budget.start()
    z=np.asarray(z,dtype=float)
    J=finite_jacobian(fun,z)
    _,sv,vh=np.linalg.svd(J,full_matrices=True)
    if sv[-1] <= 1e-10*max(1.,sv[0]):
        raise ValueError('JACOBIAN_SINGULAR')
    tangent=vh[-1]
    if previous_tangent is not None:
        if np.dot(tangent,previous_tangent)<0: tangent=-tangent
    elif tangent[-1]<0: tangent=-tangent
    predicted=z+ds*tangent
    def residual(y):
        if budget: budget.check()
        return np.append(fun(y),np.dot(y-predicted,tangent))
    fit=least_squares(residual,predicted,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    err=float(max(abs(residual(fit.x))))
    if not fit.success or err>1e-10:
        raise ValueError('ROOT_NOT_CONVERGED')
    return fit.x,tangent,{'residual':err,'singular_values':sv.tolist(),'nfev':fit.nfev,'arc_step':ds}
