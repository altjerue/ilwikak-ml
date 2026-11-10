"""Dataset loading, input encoding, trajectory-level splits and PCA compression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from . import config as cfg

#: log10 n bins farther than this below the snapshot peak are excluded from metrics
RELEVANT_DEX = 8.0


@dataclass
class Trajectories:
    u: np.ndarray  # (n, 6) unit-cube parameters
    theta: np.ndarray  # (n, 6) physical parameters
    log_n: np.ndarray  # (n, T, G)
    N_tot: np.ndarray  # (n, T)
    t_out: np.ndarray  # (T,)
    gamma: np.ndarray  # (G,)

    def __len__(self):
        return len(self.u)

    def subset(self, idx) -> "Trajectories":
        return Trajectories(self.u[idx], self.theta[idx], self.log_n[idx], self.N_tot[idx], self.t_out, self.gamma)


def load(path: Path | str, ok_only: bool = True) -> Trajectories:
    with h5py.File(path, "r") as f:
        status = f["status"][:]
        keep = status == 0 if ok_only else np.ones(len(status), bool)
        return Trajectories(
            u=f["u"][:][keep],
            theta=f["theta"][:][keep],
            log_n=f["log_n"][:][keep],
            N_tot=f["N_tot"][:][keep],
            t_out=f["t_out"][:],
            gamma=f["gamma"][:],
        )


def encode_time(t) -> np.ndarray:
    """log10 t mapped to [0, 1] over the output-time range."""
    return (np.log10(t) - cfg.LOG_T_OUT_MIN) / (cfg.LOG_T_OUT_MAX - cfg.LOG_T_OUT_MIN)


def encode_inputs(u: np.ndarray, t) -> np.ndarray:
    """Network input: unit-cube parameters + encoded time, shape (..., 7)."""
    u = np.asarray(u, dtype=np.float32)
    tt = np.broadcast_to(encode_time(t), u.shape[:-1]).astype(np.float32)
    return np.concatenate([u, tt[..., None]], axis=-1)


def flatten(tr: Trajectories):
    """(trajectory, time) pairs -> X (N, 7), Y (N, G)."""
    n, T, G = tr.log_n.shape
    u = np.repeat(tr.u, T, axis=0)
    t = np.tile(tr.t_out, n)
    return encode_inputs(u, t), tr.log_n.reshape(n * T, G).astype(np.float32)


def split_indices(n: int, val_frac: float = 0.1, seed: int = 0):
    """Random trajectory-level split: no time slice of a val trajectory is trained on."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_frac * n))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def relevant_mask(log_n_true: np.ndarray, dex: float = RELEVANT_DEX) -> np.ndarray:
    return log_n_true >= log_n_true.max(axis=-1, keepdims=True) - dex


@dataclass
class PCABasis:
    """Centered PCA of log10 n. Coefficients are scaled to unit variance."""

    mean: np.ndarray  # (G,)
    components: np.ndarray  # (k, G), orthonormal rows
    scale: np.ndarray  # (k,) std of each coefficient

    @classmethod
    def fit(cls, Y: np.ndarray, k: int | None = None, tol_dex: float = 0.005, max_rows: int = 200_000, seed: int = 0):
        """Fit on Y (N, G). If k is None, pick the smallest k whose 99th-percentile
        reconstruction error on relevant bins is below ``tol_dex``."""
        rng = np.random.default_rng(seed)
        if len(Y) > max_rows:
            Y = Y[rng.choice(len(Y), max_rows, replace=False)]
        Y = Y.astype(np.float64)
        mean = Y.mean(0)
        _, s, vt = np.linalg.svd(Y - mean, full_matrices=False)
        scale = np.maximum(s / np.sqrt(len(Y)), 1e-6 * s[0] / np.sqrt(len(Y)))
        if k is None:
            rel = relevant_mask(Y)
            k = len(s)
            for kk in range(4, len(s) + 1, 2):
                rec = (Y - mean) @ vt[:kk].T @ vt[:kk] + mean
                if np.percentile(np.abs(rec - Y)[rel], 99) < tol_dex:
                    k = kk
                    break
        return cls(mean.astype(np.float32), vt[:k].astype(np.float32), scale[:k].astype(np.float32))

    @property
    def k(self) -> int:
        return len(self.scale)

    def encode(self, Y):
        return ((Y - self.mean) @ self.components.T) / self.scale

    def decode(self, Z):
        return (Z * self.scale) @ self.components + self.mean

    def to_dict(self):
        return {"mean": self.mean, "components": self.components, "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["mean"]), np.asarray(d["components"]), np.asarray(d["scale"]))
