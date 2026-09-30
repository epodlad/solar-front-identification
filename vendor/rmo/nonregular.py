"""Synthetic coplanar Hugoniot loci; both algebraic roots are retained.

Normalized specific volume is V=v_d/v_u=rho_u/rho_d, so compression has
V<1. Source: Takahashi & Yamada arXiv:1310.2330, equations 59--61, 72.
Entropy acceptance and full-system evolutionarity are reported separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import time

from .models import State
from .physics import characteristic_speeds, scaled_inf_residual, entropy_over_cv, validate_state
from .wave_curves import CurveResult, apply_rotation


def characteristic_region(state: State, speed: float, gamma: float, tol: float = 1e-9) -> str:
    c = characteristic_speeds(state, gamma)
    v = abs(state.u[0] - speed)
    cf, ca, cs = (float(c[k]) for k in ('fast', 'alfven_n', 'slow'))
    for x, name in ((cf, '1,2'), (ca, '2,3'), (cs, '3,4')):
        if abs(v-x) <= tol * max(1., cf, v):
            return name
    return '1' if v > cf else '2' if v > ca else '3' if v > cs else '4'


def shock_checks(up: State, down: State, speed: float, gamma: float, direction: int) -> dict:
    ur, dr = characteristic_region(up, speed, gamma), characteristic_region(down, speed, gamma)
    a = np.asarray(characteristic_speeds(up, gamma)['eigenvalues']) - speed
    b = np.asarray(characteristic_speeds(down, gamma)['eigenvalues']) - speed
    # Upstream is spatially left for a left-going wave, right for a right-going wave.
    al, ar = (a, b) if direction < 0 else (b, a)
    entropy = entropy_over_cv(down, gamma) - entropy_over_cv(up, gamma)
    regular = (ur, dr) in {('1', '2'), ('3', '4')}
    return {
        'rh_scaled_inf': scaled_inf_residual(up, down, speed, gamma),
        'entropy_change_over_cv': entropy, 'entropy_admissible': entropy >= -1e-10,
        'positive': down.rho > 1e-12 and down.p > 1e-12,
        'upstream_region': ur, 'downstream_region': dr,
        'transition': ur + '_to_' + dr,
        'incoming_characteristics': int(np.count_nonzero(al > 1e-9) + np.count_nonzero(ar < -1e-9)),
        'tangent_characteristics': int(np.count_nonzero(abs(al) <= 1e-9) + np.count_nonzero(abs(ar) <= 1e-9)),
        'regular_evolutionary': regular,
        'evolutionarity_basis': 'full_7_equation_ideal_MHD_region_test; counts_alone_not_sufficient',
        'stability': 'NOT_ASSESSED',
    }


@dataclass(frozen=True)
class LocusResult:
    accepted: tuple[CurveResult, ...]
    rejected: tuple[dict, ...]
    coefficients: tuple[float, ...]


def hugoniot_at_ratio(up: State, ratio: float, gamma: float, direction: int = -1) -> LocusResult:
    """Bt_down = ratio * Bt_up; return *all* compressive entropy-allowed roots.

    ratio=0 is the switch-off shock. ratio=-1 also includes the exact
    rotational endpoint. Vanishing upstream Bt requires switch_on_shock.
    """
    if direction not in (-1, 1) or validate_state(up,gamma)!=['OK'] or not math.isfinite(ratio):
        raise ValueError('INVALID_STATE')
    norm = math.hypot(*up.B[1:])
    if norm < 1e-12 or abs(up.B[0]) < 1e-12:
        raise ValueError('ODE_EVENT_DEGENERACY')
    A, B = norm/math.sqrt(up.p), up.B[0]/math.sqrt(up.p)
    D = ratio * A
    G = gamma/(gamma-1)
    H = (gamma+1)/(gamma-1)
    aa = D/2 * (4*G + (D-A)**2 + H*(A*A-D*D)) - H*B*B*(D-A)
    bb = 2*G*(A/2*(D*D-A*A) + B*B*(D-A) - (D+A))
    cc = 2*G*A - (A*A+B*B)*(D-A)
    roots = np.roots([aa, bb, cc] if abs(aa) > 1e-14 else [bb, cc])
    accepted, rejected = [], []
    for i, root in enumerate(sorted(roots, key=lambda x: float(np.real(x)))):
        record = {'algebraic_root': i, 'V_real': float(np.real(root)), 'V_imag': float(np.imag(root)), 'Bt_ratio': ratio}
        if abs(np.imag(root)) > 1e-10:
            rejected.append({**record, 'reason': 'complex_specific_volume'}); continue
        V = float(np.real(root))
        if abs(V-1) < 1e-10 and abs(ratio+1) < 1e-12:
            rot = apply_rotation(up, math.pi, direction, gamma)
            rot.checks.update(record)
            rot.checks['limiting_structure'] = 'pi_rotation'
            rot.checks['stability'] = 'NOT_ASSESSED'
            accepted.append(rot); continue
        if not (1e-6 < V < 1.-1e-12):
            rejected.append({**record, 'reason': 'not_compressive_or_numeric_boundary'}); continue
        denom = V*D-A
        if abs(denom) < 1e-14:
            rejected.append({**record, 'reason': 'JACOBIAN_SINGULAR'}); continue
        Q = B*B*(D-A)/denom  # rho_u*(u_u-s)^2 / p_u
        pd = up.p * (1 + Q*(1-V) + .5*(A*A-D*D))
        if Q <= 0 or pd <= 1e-12:
            rejected.append({**record, 'reason': 'INVALID_STATE'}); continue
        m = -direction * math.sqrt(Q*up.rho*up.p)
        speed = up.u[0]-m/up.rho
        bt = np.array(up.B[1:])*ratio
        ut = np.array(up.u[1:]) + up.B[0]*(bt-np.array(up.B[1:]))/m
        down = State(up.rho/V, pd, (speed+m*V/up.rho, *map(float, ut)), (up.B[0], *map(float, bt)))
        checks = shock_checks(up, down, speed, gamma, direction)
        checks.update(record)
        if checks['rh_scaled_inf'] > 1e-10 or not checks['entropy_admissible']:
            rejected.append({**record, 'reason': 'RH_or_entropy', 'checks': checks}); continue
        transition = checks['transition']
        family = 'fast' if transition == '1_to_2' else 'slow' if transition == '3_to_4' else 'intermediate'
        structure = 'switch_off_shock' if ratio == 0 else 'shock' if family != 'intermediate' else 'intermediate_shock'
        accepted.append(CurveResult(down, structure, family, speed, checks))
    return LocusResult(tuple(accepted), tuple(rejected), (aa, bb, cc))


def switch_on_shock(up: State, relative_speed: float, gamma: float, direction: int = -1, angle: float = 0.) -> CurveResult:
    """Exact switch-on limiting branch, retaining its freely specified Bt angle."""
    if validate_state(up,gamma)!=['OK'] or not math.isfinite(relative_speed) or relative_speed<=0 or not math.isfinite(angle):
        raise ValueError('INVALID_STATE')
    if math.hypot(*up.B[1:]) > 1e-12 or abs(up.B[0]) < 1e-12 or direction not in (-1, 1):
        raise ValueError('INVALID_STATE')
    B2, Q = up.B[0]**2/up.p, up.rho*relative_speed**2/up.p
    V = B2/Q
    D2 = (Q-B2)/B2 * ((gamma+1)*B2-(gamma-1)*Q-2*gamma)
    if not (0 < V < 1) or D2 <= 0:
        raise ValueError('OUTSIDE_NUMERIC_DOMAIN')
    bt = math.sqrt(up.p*D2)*np.array([math.cos(angle), math.sin(angle)])
    pd = up.p*(1+Q*(1-V)-D2/2)
    m = -direction*up.rho*relative_speed
    s = up.u[0]-m/up.rho
    ut = np.array(up.u[1:])+up.B[0]*bt/m
    down = State(up.rho/V, pd, (s+m*V/up.rho, *map(float, ut)), (up.B[0], *map(float, bt)))
    checks = shock_checks(up, down, s, gamma, direction)
    checks['generated_field_angle'] = angle
    if checks['rh_scaled_inf'] > 1e-10 or not checks['entropy_admissible']:
        raise ValueError('REFERENCE_DISAGREEMENT')
    return CurveResult(down, 'switch_on_shock', 'intermediate', s, checks)


def switch_off_rarefaction(up: State, gamma: float, direction: int=-1, deadline=None) -> CurveResult:
    """Fast integral curve to Bt=0 using |Bt| as parameter, not pressure.

    Valid only when c_f^2-a^2 stays separated from zero. Any other limiting
    regime is explicitly rejected, not continued through a singularity.
    """
    b0=math.hypot(*up.B[1:])
    if b0<=1e-12 or abs(up.B[0])<=1e-12 or direction not in (-1,1):
        raise ValueError('ODE_EVENT_DEGENERACY')
    unit=np.array(up.B[1:])/b0
    entropy_constant=up.p/up.rho**gamma
    def rhs(b,y):
        if deadline and time.monotonic()>deadline: raise TimeoutError('SEARCH_DOMAIN_TRUNCATED')
        rho=y[0]
        if rho<=1e-12: raise ValueError('OUTSIDE_NUMERIC_DOMAIN')
        p=entropy_constant*rho**gamma
        s=State(rho,p,(y[1],y[2],y[3]),(up.B[0],*(b*unit)))
        c=float(characteristic_speeds(s,gamma)['fast']); a2=gamma*p/rho
        den=c*c-a2
        if den<=1e-12*max(1.,c*c): raise ValueError('ODE_EVENT_DEGENERACY')
        dr=b/den
        return [dr,direction*c*dr/rho,*(-direction*up.B[0]*unit/(rho*c))]
    fit=solve_ivp(rhs,(b0,0.),[up.rho,*up.u],method='DOP853',rtol=1e-10,atol=1e-12)
    if not fit.success: raise ValueError('ROOT_NOT_CONVERGED')
    rho,un,ut1,ut2=map(float,fit.y[:,-1])
    down=State(rho,entropy_constant*rho**gamma,(un,ut1,ut2),(up.B[0],0.,0.))
    speeds=[]
    for b,y in zip(fit.t,fit.y.T):
        s=State(y[0],entropy_constant*y[0]**gamma,tuple(y[1:]),(up.B[0],*(b*unit)))
        speeds.append(s.u[0]+direction*float(characteristic_speeds(s,gamma)['fast']))
    mono=bool(np.all(-direction*np.diff(speeds)>=-1e-9*max(1.,*map(abs,speeds))))
    checks={'isentropic':True,'entropy_change_over_cv':entropy_over_cv(down,gamma)-entropy_over_cv(up,gamma),
            'characteristic_speed_monotonic':mono,'parameter':'Bt_magnitude','nfev':fit.nfev,
            'endpoint_Bt':0.,'stability':'NOT_ASSESSED','limiting_regime':'c_f_squared_minus_a_squared_nonzero'}
    if not mono: raise ValueError('wave_speed_order')
    return CurveResult(down,'rarefaction','fast', (min(speeds[0],speeds[-1]),max(speeds[0],speeds[-1])),checks)
