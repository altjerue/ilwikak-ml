"""Verification of the Fokker-Planck problem builder against analytic solutions.

These tests pin down the coefficient convention (D = 2 D_pp,
γdot = γdot_cool - (2+q) D0 γ^(q-1)) and the solver accuracy that the
surrogate's error budget is compared against.
"""

import numpy as np
import pytest

from fp_surrogate import analytic as A
from fp_surrogate import config as cfg
from fp_surrogate import problem as P

MID = np.array([2.5, 1.0, 3.0, 4.0, -2.0, 1.8])
# Effectively no diffusion: tests the cooling + escape limit
COOL = np.array([2.5, 1.0, 4.0, 3.0, -12.0, 2.0])


def test_grids():
    assert np.allclose(cfg.GAMMA_SOLVER[:: cfg.GAMMA_REFINE], cfg.GAMMA)
    t, idx = P.solver_times()
    assert t[0] == 0.0 and np.all(np.diff(t) > 0)
    assert np.allclose(t[idx], cfg.T_OUT)


@pytest.mark.parametrize("theta", [MID, COOL, cfg.PARAM_BOUNDS[:, 0], cfg.PARAM_BOUNDS[:, 1]])
def test_injection_normalized(theta):
    Q = P.injection(theta, cfg.GAMMA_SOLVER)
    assert Q.min() >= 0
    assert P.number(Q, cfg.GAMMA_SOLVER) == pytest.approx(1.0, rel=1e-12)


def test_transport_matches_tleco_convention():
    g = cfg.GAMMA_SOLVER
    # comparisons/fokker_planck_test.py: D_0 = g^2 / (2 t_acc), D = 2 D_0, gdot = C0 g^2 - 4 D_0/g
    c0, tacc = 3.48e-11, 1e5
    gdot, D = P.transport(g, c0, 0.5 / tacc, 2.0)
    d_0 = 0.5 * g**2 / tacc
    assert np.allclose(D, 2 * d_0) and np.allclose(gdot, c0 * g**2 - 4 * d_0 / g)
    # Park & Petrosian test: D_0 = g^3, D = 2 D_0, gdot = g^2 - 3 g^2 - 2 D_0/g
    gdot, D = P.transport(g, 1.0, 1.0, 3.0)
    assert np.allclose(D, 2 * g**3) and np.allclose(gdot, g**2 - 3 * g**2 - 2 * g**3 / g)


def test_family_never_hits_cooling_only_branch():
    """Tleco falls back to fp_findif_cool when ΣD < 1e-150; our family must not."""
    lo = cfg.PARAM_BOUNDS[:, 0].copy()
    lo[cfg.PARAM_NAMES.index("q")] = cfg.PARAM_BOUNDS[-1, 0]
    _, _, D = P.coefficients(lo, cfg.GAMMA_SOLVER)
    assert D.sum() > 1e-100


def test_acc_cool_steady_state():
    """Hard-sphere acceleration balanced by cooling, no escape (Tleco test case)."""
    c0 = 3.48e-11
    tacc = 1 / (c0 * 10**4.5)
    g = np.logspace(0, np.log10(1.5e8), 400)
    t = np.concatenate([[0.0], np.logspace(0, np.log10(300 * tacc), 800)])
    gdot, D = P.transport(g, c0, 0.5 / tacc, 2.0)
    n0 = np.where((g >= 1e4) & (g <= 1e6), 1.0, 0.0)
    n0 /= np.trapezoid(n0, g)
    n = P.evolve(n0, t, np.zeros_like(g), gdot, D, g=g, t_esc=1e200, out_idx=[len(t) - 1])[-1]
    na = A.acc_cool_steady_state(g, 1.0, c0, tacc)
    m = (g > 10) & (g < 1e7)
    assert np.trapezoid(n, g) == pytest.approx(1.0, rel=1e-8)
    assert np.sqrt(np.mean(((n[m] - na[m]) / na[m]) ** 2)) < 0.02


@pytest.mark.parametrize("t_check", [5e-3, 1e-2])
def test_park_petrosian_eq59(t_check):
    g = np.logspace(0, np.log10(1.5e8), 400)
    t = np.logspace(-4, -1, 800)
    gdot, D = P.transport(g, 1.0, 1.0, 3.0)
    n0 = A.park_petrosian_eq59(1e-4, g)
    norm = np.trapezoid(n0, g)
    ns = P.evolve(n0 / norm, t, np.zeros_like(g), gdot, D, g=g, t_esc=1.0)
    i = np.argmin(np.abs(t - t_check))
    ef = A.park_petrosian_eq59(t[i] + 1e-4, g) / norm
    m = ef > ef.max() * 1e-4
    assert np.median(np.abs(np.log10(ns[i][m] / ef[m]))) < 0.03


def test_park_petrosian_matches_tleco_implementation():
    import tleco

    g = np.logspace(0, 8, 200)
    ref = np.asarray(tleco.eq_59_park1995(1e-2, g))
    ok = ref > 1e-90
    assert np.allclose(A.park_petrosian_eq59(1e-2, g)[ok], ref[ok], rtol=1e-3)


def test_number_budget():
    """Unit injection + escape: N(t) = 1 - exp(-t), for an in-family trajectory."""
    g = cfg.GAMMA_SOLVER
    Q, gdot, D = P.coefficients(MID, g)
    t, idx = P.solver_times()
    n = P.evolve(np.zeros_like(g), t, Q, gdot, D, g=g, out_idx=idx)
    N = P.number(n, g)
    assert np.max(np.abs(N / A.number_budget(cfg.T_OUT) - 1)) < 0.01


def test_cooling_escape_steady_state():
    g = cfg.GAMMA_SOLVER
    Q, gdot, D = P.coefficients(COOL, g)
    t, idx = P.solver_times()
    n = P.evolve(np.zeros_like(g), t, Q, gdot, D, g=g, out_idx=[idx[-1]])[-1]
    na = A.cooling_escape_steady_state(g, Q, gc=1e3)
    m = (g > 30) & (g < 3e4)
    assert np.median(np.abs(np.log10(n[m] / na[m]))) < 0.03
    # Spectral indices: p below the cooling break, p + 1 above it
    p = COOL[0]
    assert A.local_index(g, n, 30, 300) == pytest.approx(p, abs=0.1)
    assert A.local_index(g, n, 3e3, 1e4) == pytest.approx(p + 1, abs=0.15)


def test_solve_trajectory_shape_and_sign():
    n = P.solve_trajectory(MID)
    assert n.shape == (cfg.N_TOUT, cfg.N_GAMMA)
    assert np.all(np.isfinite(n)) and n.min() >= 0


@pytest.mark.xfail(reason="Tleco fp_findif_cool uses the opposite γdot sign to fp_findif_difu", strict=True)
def test_tleco_cooling_only_branch_sign():
    g = cfg.GAMMA_SOLVER
    Q, gdot, _ = P.coefficients(COOL, g)
    t, idx = P.solver_times()
    n = P.evolve(np.zeros_like(g), t, Q, gdot, np.zeros_like(g), g=g, out_idx=[idx[-1]])[-1]
    assert np.all(np.isfinite(n)) and n.min() >= 0 and n.max() < 1e3
