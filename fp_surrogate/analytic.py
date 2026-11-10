"""Analytic reference solutions used to verify the solver and the surrogate."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import iv


def number_budget(t, t_esc: float = 1.0, rate: float = 1.0):
    """N(t) for unit-rate injection from N(0) = 0 with escape only.

    dN/dt = rate - N/t_esc  =>  N = rate t_esc (1 - exp(-t/t_esc)).
    Cooling and diffusion move particles in γ but conserve their number
    (zero-flux boundaries), so this holds for the whole problem family.
    """
    return rate * t_esc * (1.0 - np.exp(-np.asarray(t) / t_esc))


def cooling_escape_steady_state(g, Q, gc: float, t_esc: float = 1.0):
    """Steady state of dn/dt = d(b γ^2 n)/dγ + Q - n/t_esc, with b = 1/(γc t_esc).

    n(γ) = 1/(b γ^2) ∫_γ^∞ Q(γ') exp[-γc (1/γ - 1/γ')] dγ'

    (γc (1/γ - 1/γ') is the cooling time from γ' to γ in units of t_esc).
    Evaluated by trapezoid quadrature on the grid ``g``.
    """
    b = 1.0 / (gc * t_esc)
    n = np.zeros_like(g)
    for i, gi in enumerate(g):
        gp, qp = g[i:], Q[i:]
        if len(gp) < 2:
            continue
        w = np.exp(-gc * (1.0 / gi - 1.0 / gp))
        n[i] = np.trapezoid(qp * w, gp) / (b * gi**2)
    return n


def acc_cool_steady_state(g, n_total: float, c0: float, t_acc: float):
    """Tleco test case: hard-sphere acceleration (D_pp = γ^2/(2 t_acc)) + cooling c0 γ^2, no escape.

    The zero-flux steady state is n ∝ γ^2 exp(-2 c0 t_acc (γ - 1)),
    normalized here to ``n_total`` particles.
    """
    shape = g**2 * np.exp(-2 * c0 * t_acc * (g - 1))
    return n_total * shape / np.trapezoid(shape, g)


def park_petrosian_eq59(t, g):
    """Park & Petrosian (1995) eq. 59 Green's function (q = 3, a = 1, θ = 1, x0 ≈ 100).

    Mirrors ``tleco.eq_59_park1995`` / ``comparisons/dependencies/fp_test.py``.
    """
    q, a, theta = 3.0, 1.0, 1.0
    x = np.sqrt(g**2 - 1.0)
    x0 = np.sqrt(100.0**2 - 1.0)
    alpha = (2 - q) / 2
    tau = t
    with np.errstate(all="ignore"):
        gfa = 1 / abs(alpha) / (2 * tau)
        gfb = x ** ((1 - q + a) / 2) * x0 ** ((1 - q - a) / 2)
        order = abs((q - 1 + a) / (2 * alpha))
        gfc = iv(order, x**alpha * x0**alpha / (2 * tau * alpha**2))
        gfd = np.exp(-(x ** (2 * alpha) + x0 ** (2 * alpha)) / (4 * alpha**2 * tau))
        out = gfa * gfb * gfc * gfd * np.exp(-theta * tau) * 4 * np.pi
    return np.nan_to_num(out, nan=0.0, posinf=0.0)


def local_index(g, n, g_lo, g_hi):
    """Least-squares slope of log n vs log γ on [g_lo, g_hi] (returns -slope)."""
    m = (g >= g_lo) & (g <= g_hi) & (n > 0)
    if m.sum() < 3:
        return np.nan
    return -np.polyfit(np.log10(g[m]), np.log10(n[m]), 1)[0]


__all__ = [
    "number_budget",
    "cooling_escape_steady_state",
    "acc_cool_steady_state",
    "park_petrosian_eq59",
    "local_index",
    "cumulative_trapezoid",
    "iv",
]
