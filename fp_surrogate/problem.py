"""Build Fokker-Planck coefficient arrays from physical parameters and run Tleco.

Tleco's ``fp_findif_difu`` is a Chang-Cooper implicit scheme for

    dn/dt = d/dγ [ (1/2) d(D n)/dγ + γdot n ] + Q - n / t_esc ,

where a positive ``γdot`` means energy loss. Momentum diffusion by
turbulence with coefficient D_pp = D0 γ^q obeys (ultra-relativistic limit)

    dn/dt = d/dγ [ D_pp γ^2 d(n/γ^2)/dγ + γdot_cool n ] + ...

Matching the two forms gives the arrays the solver expects:

    D     = 2 D_pp
    γdot  = γdot_cool - (2 + q) D0 γ^(q-1)

For q = 2 this is the convention of Tleco's own test
(``comparisons/fokker_planck_test.py``: D = 2 D_0, γdot = C0 γ^2 - 4 D_0/γ).
"""

from __future__ import annotations

import numpy as np
import tleco

from . import config as cfg


def unpack(theta) -> dict:
    """Map a parameter vector (in PARAM_NAMES order) to a dict."""
    return dict(zip(cfg.PARAM_NAMES, np.asarray(theta, dtype=float)))


def injection(theta, g=cfg.GAMMA) -> np.ndarray:
    """Power-law injection Q ∝ γ^-p on [γ1, γ2], normalized to ∫Q dγ = 1.

    The normalization uses the same trapezoid weights as ``number`` so the
    discrete injected number is exactly one per unit time.
    """
    th = unpack(theta)
    g1 = 10 ** th["log_g1"]
    g2 = min(g1 * 10 ** th["log_g2_g1"], cfg.GAMMA2_CAP)
    q = np.where((g >= g1) & (g <= g2), g ** (-th["p"]), 0.0)
    if not q.any():  # injection narrower than a bin: put it in the nearest bin
        q[np.argmin(np.abs(np.log(g / g1)))] = 1.0
    return q / number(q, g)


def transport(g, b: float, d0: float, q: float):
    """Solver arrays (gdot, D) for cooling b γ^2 plus diffusion D_pp = d0 γ^q."""
    d_pp = d0 * g**q
    gdot = b * g**2 - (2.0 + q) * d0 * g ** (q - 1.0)
    return gdot, 2.0 * d_pp


def coefficients(theta, g=cfg.GAMMA):
    """Return (Q, gdot, D) arrays for ``fp_findif_difu`` (t_esc = 1)."""
    th = unpack(theta)
    b = 10 ** (-th["log_gc"])  # γdot_cool = b γ^2 with b = 1 / (γc t_esc)
    gdot, D = transport(g, b, 10 ** th["log_D0"], th["q"])
    return injection(theta, g), gdot, D


def number(n, g=cfg.GAMMA) -> float | np.ndarray:
    """Particle number ∫ n dγ (trapezoid), along the last axis."""
    return np.trapezoid(n, g, axis=-1)


def solver_times(t_out=cfg.T_OUT, solver: cfg.SolverConfig = cfg.DEFAULT_SOLVER):
    """Solver time grid (starting at 0) containing every output time exactly.

    Returns (t, out_idx) with t[out_idx] == t_out.
    """
    log_out = np.log10(t_out)
    dlog = np.diff(log_out).min() / solver.substeps
    pre = np.arange(log_out[0] - dlog, solver.log_t_start - 1e-12, -dlog)[::-1]
    segs = [pre]
    for a, b in zip(log_out[:-1], log_out[1:]):
        m = max(1, int(round((b - a) / dlog)))
        segs.append(np.linspace(a, b, m + 1)[:-1])
    segs.append([log_out[-1]])
    log_t = np.concatenate(segs)
    t = np.concatenate([[0.0], 10**log_t])
    out_idx = np.searchsorted(t, t_out)
    assert np.allclose(t[out_idx], t_out)
    return t, out_idx


def evolve(n0, t, Q, gdot, D, g=cfg.GAMMA, t_esc=1.0, out_idx=None):
    """Step ``fp_findif_difu`` along the time grid ``t``.

    Coefficients are constant in time. Returns snapshots at ``out_idx``
    (or every step when ``out_idx`` is None).
    """
    n = np.asarray(n0, dtype=float).copy()
    keep = set(range(len(t))) if out_idx is None else set(int(i) for i in out_idx)
    snaps = []
    if 0 in keep:
        snaps.append(n.copy())
    for i in range(1, len(t)):
        n = np.asarray(
            tleco.fp_findif_difu(t[i] - t[i - 1], g, n, gdot, D, Q, t_esc, 1.0, False)
        )
        if i in keep:
            snaps.append(n.copy())
    return np.array(snaps)


def solve_trajectory(
    theta,
    t_out=cfg.T_OUT,
    solver: cfg.SolverConfig = cfg.DEFAULT_SOLVER,
    g_solver=cfg.GAMMA_SOLVER,
    refine: int = cfg.GAMMA_REFINE,
):
    """Solve from n(γ, 0) = 0 with continuous injection; return n at ``t_out``.

    The solver runs on ``g_solver`` and the result is subsampled to the
    output grid (every ``refine``-th point), shape (len(t_out), N_GAMMA).
    """
    Q, gdot, D = coefficients(theta, g_solver)
    t, out_idx = solver_times(t_out, solver)
    n = evolve(np.zeros_like(g_solver), t, Q, gdot, D, g=g_solver, out_idx=out_idx)
    return n[:, ::refine]


def to_log(n, floor_dex: float = cfg.LOG_FLOOR_DEX) -> np.ndarray:
    """log10 n with a floor ``floor_dex`` below each snapshot's maximum."""
    n = np.asarray(n, dtype=float)
    nmax = np.max(n, axis=-1, keepdims=True)
    floor = np.maximum(nmax, 1e-300) * 10.0 ** (-floor_dex)
    return np.log10(np.maximum(n, floor))


def gamma_eq(theta) -> float:
    """Acceleration-cooling balance: (2+q) D0 γ^(q-1) = γ^2 / γc."""
    th = unpack(theta)
    return ((2 + th["q"]) * 10 ** th["log_D0"] * 10 ** th["log_gc"]) ** (1 / (3 - th["q"]))
