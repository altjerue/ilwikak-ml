"""Fixed grids, parameter bounds and paths for the Fokker-Planck surrogate.

All quantities are dimensionless: time is measured in units of the escape
time t_esc (t_esc = 1) and the injection rate is normalized to unity
(integral of Q over gamma = 1). The Fokker-Planck equation is linear in
(n, Q), so any physical normalization is recovered by rescaling.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fp_surrogate"

# ---------------------------------------------------------------- grids
N_GAMMA = 128
GAMMA_MIN, GAMMA_MAX = 1.0, 1e8
GAMMA = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA)

# The reference solver runs on a finer grid that contains GAMMA exactly
# (every GAMMA_REFINE-th point). Tleco's scheme is first order in Δγ, so a 4x
# refinement cuts the truncation error vs. the continuum solution ~4x
# (median ~0.08 dex -> ~0.02 dex on the cooling steady state).
GAMMA_REFINE = 4
N_GAMMA_SOLVER = GAMMA_REFINE * (N_GAMMA - 1) + 1
GAMMA_SOLVER = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA_SOLVER)

# Output (snapshot) times, in units of t_esc
N_TOUT = 32
LOG_T_OUT_MIN, LOG_T_OUT_MAX = -3.0, 2.0
T_OUT = np.logspace(LOG_T_OUT_MIN, LOG_T_OUT_MAX, N_TOUT)

# Floor applied to log10 n relative to each snapshot's maximum
LOG_FLOOR_DEX = 15.0

# ---------------------------------------------------------------- parameters
# name, lower, upper. All sampled uniformly in the listed coordinate.
PARAM_NAMES = ["p", "log_g1", "log_g2_g1", "log_gc", "log_D0", "q"]
PARAM_LABELS = {
    "p": r"$p$",
    "log_g1": r"$\log_{10}\gamma_1$",
    "log_g2_g1": r"$\log_{10}(\gamma_2/\gamma_1)$",
    "log_gc": r"$\log_{10}\gamma_c$",
    "log_D0": r"$\log_{10}(D_0 t_\mathrm{esc})$",
    "q": r"$q$",
}
PARAM_BOUNDS = np.array(
    [
        [1.5, 3.5],  # p: injection power-law index, Q ∝ γ^-p on [γ1, γ2]
        [0.0, 3.0],  # log10 γ1: injection lower cutoff
        [1.0, 5.0],  # log10(γ2/γ1): injection width (γ2 capped at GAMMA2_CAP)
        [1.0, 6.0],  # log10 γc: cooling break, γdot_cool = γ^2 / (γc t_esc)
        [-4.0, 0.0],  # log10(D0 t_esc): stochastic acceleration, D_pp = D0 γ^q
        [1.5, 2.0],  # q: turbulence spectrum index (5/3 Kolmogorov, 2 hard-sphere)
    ]
)
GAMMA2_CAP = 1e7

N_PARAMS = len(PARAM_NAMES)


@dataclass(frozen=True)
class SolverConfig:
    """Time-stepping resolution of the reference solver.

    The solver grid contains every output time exactly. Between consecutive
    output times there are ``substeps`` log-uniform steps; before the first
    output time the grid starts at ``log_t_start`` with the same log spacing.
    """

    substeps: int = 64
    log_t_start: float = -6.0


DEFAULT_SOLVER = SolverConfig()

# Sobol design sizes and seeds
N_TRAIN_POOL_LOG2 = 14  # train + validation trajectories
N_TEST_LOG2 = 11  # independent test trajectories
SEED_TRAIN_POOL = 20261110
SEED_TEST = 777
