"""Generate Fokker-Planck trajectories over a scrambled Sobol design.

Usage
-----
    python -m fp_surrogate.generate --split train --out data/fp_surrogate/train.h5
    python -m fp_surrogate.generate --split test  --out data/fp_surrogate/test.h5
    python -m fp_surrogate.generate --n 64 --out data/fp_surrogate/smoke.h5

The run is resumable: rows whose ``status`` is already set are skipped.

HDF5 layout
-----------
    u          (n, 6)       Sobol point in the unit cube
    theta      (n, 6)       physical parameters (config.PARAM_NAMES order)
    gamma      (128,)       output γ grid
    t_out      (32,)        output times / t_esc
    log_n      (n, 32, 128) log10 n, floored LOG_FLOOR_DEX below each snapshot max
    N_tot      (n, 32)      particle number on the solver grid
    status     (n,)         -1 pending, 0 ok, 1 non-finite, 2 negative, 3 exception
    runtime_s  (n,)
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from . import config as cfg
from . import problem as P

STATUS_PENDING, STATUS_OK, STATUS_NONFINITE, STATUS_NEGATIVE, STATUS_ERROR = -1, 0, 1, 2, 3


def sobol_design(log2_n: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points in the unit cube, shape (2**log2_n, N_PARAMS)."""
    return qmc.Sobol(d=cfg.N_PARAMS, scramble=True, seed=seed).random_base2(log2_n)


def to_theta(u: np.ndarray) -> np.ndarray:
    lo, hi = cfg.PARAM_BOUNDS.T
    return lo + (hi - lo) * np.asarray(u)


def to_unit(theta: np.ndarray) -> np.ndarray:
    lo, hi = cfg.PARAM_BOUNDS.T
    return (np.asarray(theta) - lo) / (hi - lo)


def _run_one(args):
    i, theta = args
    t0 = time.perf_counter()
    log_n = np.full((cfg.N_TOUT, cfg.N_GAMMA), np.nan, dtype=np.float32)
    N_tot = np.full(cfg.N_TOUT, np.nan)
    try:
        Q, gdot, D = P.coefficients(theta, cfg.GAMMA_SOLVER)
        t, idx = P.solver_times()
        n = P.evolve(np.zeros_like(cfg.GAMMA_SOLVER), t, Q, gdot, D, g=cfg.GAMMA_SOLVER, out_idx=idx)
        if not np.all(np.isfinite(n)):
            status = STATUS_NONFINITE
        elif n.min() < -1e-12 * n.max():
            status = STATUS_NEGATIVE
        else:
            status = STATUS_OK
            n = np.clip(n, 0.0, None)
            N_tot = P.number(n, cfg.GAMMA_SOLVER)
            log_n = P.to_log(n[:, :: cfg.GAMMA_REFINE]).astype(np.float32)
    except Exception:  # noqa: BLE001 - recorded as a status code
        status = STATUS_ERROR
    return i, log_n, N_tot, status, time.perf_counter() - t0


def _create(path: Path, u: np.ndarray):
    n = len(u)
    with h5py.File(path, "w") as f:
        f["u"] = u
        f["theta"] = to_theta(u)
        f["gamma"] = cfg.GAMMA
        f["t_out"] = cfg.T_OUT
        f.create_dataset("log_n", (n, cfg.N_TOUT, cfg.N_GAMMA), dtype="f4", chunks=(64, cfg.N_TOUT, cfg.N_GAMMA))
        f.create_dataset("N_tot", (n, cfg.N_TOUT), dtype="f8")
        f["status"] = np.full(n, STATUS_PENDING, dtype="i1")
        f["runtime_s"] = np.zeros(n)
        f.attrs["param_names"] = cfg.PARAM_NAMES
        f.attrs["param_bounds"] = cfg.PARAM_BOUNDS
        f.attrs["solver_substeps"] = cfg.DEFAULT_SOLVER.substeps
        f.attrs["solver_n_gamma"] = cfg.N_GAMMA_SOLVER
        f.attrs["log_floor_dex"] = cfg.LOG_FLOOR_DEX


def generate(out: Path, u: np.ndarray, workers: int, flush_every: int = 256):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        _create(out, u)
    with h5py.File(out, "r") as f:
        if not np.allclose(f["u"][:], u):
            raise ValueError(f"{out} exists with a different design; delete it or choose another path")
        status = f["status"][:]
        theta = f["theta"][:]
    todo = np.flatnonzero(status == STATUS_PENDING)
    print(f"{out}: {len(u)} trajectories, {len(todo)} pending, {workers} workers")
    if len(todo) == 0:
        return
    t0 = time.perf_counter()
    buf = []
    with Pool(workers) as pool, h5py.File(out, "a") as f:
        for k, res in enumerate(pool.imap_unordered(_run_one, [(i, theta[i]) for i in todo], chunksize=4), 1):
            buf.append(res)
            if len(buf) >= flush_every or k == len(todo):
                for i, log_n, N_tot, st, rt in buf:
                    f["log_n"][i] = log_n
                    f["N_tot"][i] = N_tot
                    f["status"][i] = st
                    f["runtime_s"][i] = rt
                f.flush()
                buf = []
                el = time.perf_counter() - t0
                print(f"  {k}/{len(todo)} done, {el:.0f}s elapsed, ETA {el / k * (len(todo) - k):.0f}s", flush=True)
        st = f["status"][:]
    print("status counts:", {int(s): int((st == s).sum()) for s in np.unique(st)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n", type=int, default=None, help="smoke run: first n points of the train design")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.split == "test":
        u = sobol_design(cfg.N_TEST_LOG2, cfg.SEED_TEST)
    else:
        u = sobol_design(cfg.N_TRAIN_POOL_LOG2, cfg.SEED_TRAIN_POOL)
    if a.n is not None:
        u = u[: a.n]
    generate(a.out, u, a.workers)


if __name__ == "__main__":
    main()
