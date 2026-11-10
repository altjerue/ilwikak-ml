"""Evaluate a trained surrogate on the held-out test trajectories.

Usage
-----
    python -m fp_surrogate.evaluate --tag pca_main

Writes ``data/fp_surrogate/eval/<tag>.json`` (summary) and ``<tag>.npz``
(per-sample errors and diagnostics used by the notebook).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import analytic as A
from . import config as cfg
from . import problem as P
from .data import Trajectories, flatten, load, relevant_mask
from .model import FPSurrogate
from .train import MODEL_DIR, batched_predict, error_summary

EVAL_DIR = cfg.DATA_DIR / "eval"


def predict_trajectories(model: FPSurrogate, tr: Trajectories, device="cpu") -> np.ndarray:
    X, _ = flatten(tr)
    return batched_predict(model.to(device), X, device).reshape(tr.log_n.shape)


def sample_errors(pred, true):
    """Median and max |Δ log n| over relevant bins, per (trajectory, time)."""
    e = np.where(relevant_mask(true), np.abs(pred - true), np.nan)
    return np.nanmedian(e, axis=-1), np.nanmax(e, axis=-1)


def number_on_grid(log_n, g=cfg.GAMMA):
    return np.trapezoid(10.0**log_n, g, axis=-1)


def energy_peak(log_n, g=cfg.GAMMA):
    """log10 γ at the maximum of γ^2 n (energy per log γ)."""
    return np.log10(g)[np.argmax(log_n + 2 * np.log10(g), axis=-1)]


def index_check(pred, tr: Trajectories, t_index=-1, min_span_dex=0.5):
    """Late-time spectral indices below/above γc, for cooling-dominated runs.

    Returns rows (traj, which, true_index, pred_index, expected) where
    which = 0 (below γc, expect p) or 1 (above γc, expect p+1).
    """
    rows = []
    g = tr.gamma
    for i, th in enumerate(tr.theta):
        d = P.unpack(th)
        if d["log_D0"] > -3.0:
            continue  # acceleration matters; no simple analytic index
        g1 = 10 ** d["log_g1"]
        g2 = min(g1 * 10 ** d["log_g2_g1"], cfg.GAMMA2_CAP)
        gc = 10 ** d["log_gc"]
        windows = [(0, 3 * g1, gc / 3, d["p"]), (1, max(3 * gc, 3 * g1), g2 / 10, d["p"] + 1)]
        for which, lo, hi, expect in windows:
            if np.log10(hi / lo) < min_span_dex:
                continue
            nt = 10 ** tr.log_n[i, t_index]
            npred = 10 ** pred[i, t_index]
            rows.append((i, which, A.local_index(g, nt, lo, hi), A.local_index(g, npred, lo, hi), expect))
    return np.array(rows, dtype=float).reshape(-1, 5)


def benchmark(model: FPSurrogate, theta, n_rep=20):
    """Wall time: full solver trajectory (0 -> 100 t_esc, 32 snapshots) vs surrogate."""
    t0 = time.perf_counter()
    for th in theta[:n_rep]:
        P.solve_trajectory(th)
    solver = (time.perf_counter() - t0) / n_rep
    model = model.cpu()
    model.predict(theta[:1], cfg.T_OUT[-1])  # warm-up
    t0 = time.perf_counter()
    for th in theta[:n_rep]:
        model.predict(th, 1.0)  # one (θ, t) snapshot
    single = (time.perf_counter() - t0) / n_rep
    t0 = time.perf_counter()
    for th in theta[:n_rep]:
        model.predict(np.repeat(th[None], cfg.N_TOUT, 0), cfg.T_OUT)  # all 32 snapshots
    traj = (time.perf_counter() - t0) / n_rep
    th_b = np.repeat(theta[:1], 10_000, axis=0)
    t0 = time.perf_counter()
    model.predict(th_b, np.full(len(th_b), 1.0))
    batch = (time.perf_counter() - t0) / len(th_b)
    return {"solver_traj_s": solver, "surrogate_single_s": single, "surrogate_traj_s": traj,
            "surrogate_batched_per_sample_s": batch}


def evaluate(tag: str, test_file=cfg.DATA_DIR / "test.h5", out_dir=EVAL_DIR, bench=True) -> dict:
    model = FPSurrogate.load(MODEL_DIR / f"{tag}.pt")
    te = load(test_file)
    pred = predict_trajectories(model, te)
    med, mx = sample_errors(pred, te.log_n)
    summary = {f"test_{k}": v for k, v in error_summary(pred.reshape(-1, cfg.N_GAMMA), te.log_n.reshape(-1, cfg.N_GAMMA)).items()}

    # Number budget: surrogate vs solver, both integrated on the 128-bin output grid
    N_pred, N_true = number_on_grid(pred), number_on_grid(te.log_n)
    N_ana = A.number_budget(te.t_out)
    summary["number_rel_err_pred_vs_solver_median"] = float(np.median(np.abs(N_pred / N_true - 1)))
    summary["number_rel_err_pred_vs_solver_p95"] = float(np.percentile(np.abs(N_pred / N_true - 1), 95))
    summary["number_rel_err_solver_fine_vs_analytic_p95"] = float(np.percentile(np.abs(te.N_tot / N_ana - 1), 95))

    # Energy peak position
    dpk = energy_peak(pred) - energy_peak(te.log_n)
    summary["energy_peak_abs_dex_median"] = float(np.median(np.abs(dpk)))
    summary["energy_peak_abs_dex_p95"] = float(np.percentile(np.abs(dpk), 95))

    idx = index_check(pred, te)
    if len(idx):
        ok = np.isfinite(idx[:, 2]) & np.isfinite(idx[:, 3])
        summary["index_n_checks"] = int(ok.sum())
        summary["index_abs_err_pred_vs_solver_median"] = float(np.median(np.abs(idx[ok, 3] - idx[ok, 2])))
        summary["index_abs_err_solver_vs_expected_median"] = float(np.median(np.abs(idx[ok, 2] - idx[ok, 4])))

    if bench:
        summary.update(benchmark(model, te.theta))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / f"{tag}.npz", pred=pred.astype(np.float32), med=med, max=mx,
                        N_pred=N_pred, N_true=N_true, dpk=dpk, index=idx)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-file", type=Path, default=cfg.DATA_DIR / "test.h5")
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(evaluate(a.tag, a.test_file, bench=not a.no_bench), indent=1))


if __name__ == "__main__":
    main()
