"""Train the Fokker-Planck surrogate.

Usage
-----
    python -m fp_surrogate.train --head pca --tag pca_main
    python -m fp_surrogate.train --head direct --tag direct_main
    python -m fp_surrogate.train --head pca --n-train-traj 1024 --tag lc_1024

Runs are logged to MLflow (``sqlite:///mlflow.db`` in the repo root; browse
with ``mlflow ui --backend-store-uri sqlite:///mlflow.db``). The checkpoint
goes to ``data/fp_surrogate/models/<tag>.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from . import config as cfg
from .data import PCABasis, flatten, load, relevant_mask, split_indices
from .model import FPSurrogate

MODEL_DIR = cfg.DATA_DIR / "models"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def loss_weights(Y: torch.Tensor, mode: str, mask_dex: float, w_out: float) -> torch.Tensor | None:
    """Per-bin loss weights. ``masked``: 1 within ``mask_dex`` of each snapshot peak, ``w_out`` elsewhere."""
    if mode == "all":
        return None
    near = Y >= Y.max(dim=-1, keepdim=True).values - mask_dex
    return torch.where(near, 1.0, w_out)


def weighted_mse(pred, Y, w):
    if w is None:
        return torch.mean((pred - Y) ** 2)
    return torch.sum(w * (pred - Y) ** 2) / torch.sum(w)


@torch.no_grad()
def batched_predict(model, X, device, bs=65536):
    out = [model(torch.from_numpy(X[i : i + bs]).to(device)).cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def error_summary(pred: np.ndarray, true: np.ndarray) -> dict:
    """Errors in dex on relevant bins (within RELEVANT_DEX of each snapshot peak)."""
    err = np.abs(pred - true)
    rel = relevant_mask(true)
    e_rel = np.where(rel, err, np.nan)
    per_sample_max = np.nanmax(e_rel, axis=1)
    return {
        "median_abs_dex": float(np.nanmedian(e_rel)),
        "p95_abs_dex": float(np.nanpercentile(e_rel, 95)),
        "median_sample_max_dex": float(np.median(per_sample_max)),
        "p95_sample_max_dex": float(np.percentile(per_sample_max, 95)),
        "frac_sample_max_lt_0p05": float(np.mean(per_sample_max < 0.05)),
        "rmse_all_bins": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    pool = load(args.train_file)
    tr_idx, va_idx = split_indices(len(pool), args.val_frac, seed=args.seed)
    if args.n_train_traj:
        tr_idx = np.random.default_rng(args.seed).choice(tr_idx, args.n_train_traj, replace=False)
    Xtr, Ytr = flatten(pool.subset(tr_idx))
    Xva, Yva = flatten(pool.subset(va_idx))
    print(f"train {len(tr_idx)} traj / {len(Xtr)} samples, val {len(va_idx)} traj / {len(Xva)} samples, device {device}")

    if args.head == "pca":
        basis = PCABasis.fit(Ytr, k=args.n_pca, tol_dex=args.pca_tol)
        model = FPSurrogate("pca", Ytr.shape[1], args.width, args.depth, basis=basis)
        rec = basis.decode(basis.encode(Yva))
        pca_floor = error_summary(rec, Yva)
        print(f"PCA k={basis.k}; val reconstruction floor: {pca_floor}")
    else:
        model = FPSurrogate("direct", Ytr.shape[1], args.width, args.depth,
                            y_mean=Ytr.mean(0), y_std=np.maximum(Ytr.std(0), 1e-3))
        pca_floor = None
    model.to(device)

    Xtr_t, Ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)
    n = len(Xtr_t)
    steps_per_epoch = math.ceil(n / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(0.02 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )

    mlflow.set_tracking_uri(f"sqlite:///{cfg.REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fp_surrogate")
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    params.update(n_train_traj_used=len(tr_idx), n_params=sum(p.numel() for p in model.parameters()),
                  device=device, n_pca=getattr(model.net[-1], "out_features", None) if args.head == "pca" else None)

    best, best_state, bad, history = math.inf, None, 0, []
    t0 = time.perf_counter()
    with mlflow.start_run(run_name=args.tag):
        mlflow.log_params(params)
        if pca_floor:
            mlflow.log_metrics({f"pca_floor_{k}": v for k, v in pca_floor.items()})
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            run_loss = 0.0
            for i in range(steps_per_epoch):
                b = perm[i * args.batch : (i + 1) * args.batch]
                yb = Ytr_t[b]
                loss = weighted_mse(model(Xtr_t[b]), yb, loss_weights(yb, args.loss, args.mask_dex, args.w_out))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item()
            model.eval()
            pva = batched_predict(model, Xva, device)
            Yva_t = torch.from_numpy(Yva)
            val_mse = float(weighted_mse(torch.from_numpy(pva), Yva_t, loss_weights(Yva_t, args.loss, args.mask_dex, args.w_out)))
            m = {"train_mse": run_loss / steps_per_epoch, "val_mse": val_mse, "lr": sched.get_last_lr()[0]}
            if ep % args.eval_every == 0 or ep == args.epochs - 1:
                m.update({f"val_{k}": v for k, v in error_summary(pva, Yva).items()})
            mlflow.log_metrics(m, step=ep)
            history.append(m)
            if val_mse < best:
                best, bad = val_mse, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"ep {ep:4d}  train {m['train_mse']:.3e}  val {val_mse:.3e}  "
                      f"med {m.get('val_median_abs_dex', float('nan')):.4f}  {time.perf_counter() - t0:.0f}s", flush=True)
            if bad >= args.patience:
                print(f"early stop at epoch {ep}")
                break

        model.load_state_dict(best_state)
        model.eval()
        final = {f"best_val_{k}": v for k, v in error_summary(batched_predict(model, Xva, device), Yva).items()}
        final["train_time_s"] = time.perf_counter() - t0
        final["epochs_run"] = len(history)
        mlflow.log_metrics(final)
        out = MODEL_DIR / f"{args.tag}.pt"
        model.cpu().save(out, meta={"args": params, "final": final, "val_idx": va_idx.tolist(),
                                    "train_idx": np.asarray(tr_idx).tolist(), "pca_floor": pca_floor})
        (MODEL_DIR / f"{args.tag}_history.json").write_text(json.dumps(history))
        mlflow.log_artifact(str(out))
        print(f"saved {out}\n{json.dumps(final, indent=1)}")
    return final


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-file", type=Path, default=cfg.DATA_DIR / "train.h5")
    ap.add_argument("--head", choices=["pca", "direct"], default="pca")
    ap.add_argument("--tag", default="pca_main")
    ap.add_argument("--n-train-traj", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--loss", choices=["all", "masked"], default="masked")
    ap.add_argument("--mask-dex", type=float, default=10.0)
    ap.add_argument("--w-out", type=float, default=0.01)
    ap.add_argument("--n-pca", type=int, default=None)
    ap.add_argument("--pca-tol", type=float, default=0.005)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse())
