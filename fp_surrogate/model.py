"""MLP surrogate: (θ, t) -> log10 n(γ, t).

Two output heads share the same trunk and the same loss (MSE in log10 n):

* ``pca``    — the network predicts unit-variance PCA coefficients, decoded to
               log10 n by a fixed linear layer (mean + Σ z_k s_k v_k).
* ``direct`` — the network predicts all G bins, as mean + std ⊙ output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import PCABasis, encode_inputs
from .generate import to_unit


def mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class FPSurrogate(nn.Module):
    def __init__(self, head: str, n_gamma: int, width: int = 256, depth: int = 4, basis: PCABasis | None = None,
                 y_mean=None, y_std=None):
        super().__init__()
        self.head, self.width, self.depth = head, width, depth
        if head == "pca":
            assert basis is not None
            self.net = mlp(7, basis.k, width, depth)
            self.register_buffer("dec_w", torch.tensor(basis.scale[:, None] * basis.components))  # (k, G)
            self.register_buffer("dec_b", torch.tensor(basis.mean))
        elif head == "direct":
            self.net = mlp(7, n_gamma, width, depth)
            self.register_buffer("dec_w", torch.tensor(np.asarray(y_std, np.float32)))  # (G,)
            self.register_buffer("dec_b", torch.tensor(np.asarray(y_mean, np.float32)))
        else:
            raise ValueError(head)

    def forward(self, x):
        z = self.net(x)
        if self.head == "pca":
            return z @ self.dec_w + self.dec_b
        return z * self.dec_w + self.dec_b

    # ------------------------------------------------------------ persistence
    def save(self, path: Path, meta: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head": self.head,
                "width": self.width,
                "depth": self.depth,
                "n_gamma": int(self.dec_b.shape[0]),
                "n_out": int(self.net[-1].out_features),
                "state_dict": self.state_dict(),
                "meta": meta or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, map_location="cpu") -> "FPSurrogate":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        G, k = ck["n_gamma"], ck["n_out"]
        if ck["head"] == "pca":
            dummy = PCABasis(np.zeros(G, np.float32), np.zeros((k, G), np.float32), np.ones(k, np.float32))
            m = cls("pca", G, ck["width"], ck["depth"], basis=dummy)
        else:
            m = cls("direct", G, ck["width"], ck["depth"], y_mean=np.zeros(G), y_std=np.ones(G))
        m.load_state_dict(ck["state_dict"])
        m.meta = ck["meta"]
        return m.eval()

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, theta, t, device="cpu") -> np.ndarray:
        """log10 n for physical parameters ``theta`` (..., 6) at times ``t`` (broadcastable)."""
        x = encode_inputs(to_unit(np.asarray(theta, float)), t)
        shape = x.shape[:-1]
        out = self.to(device)(torch.from_numpy(x.reshape(-1, 7)).to(device))
        return out.cpu().numpy().reshape(*shape, -1)
