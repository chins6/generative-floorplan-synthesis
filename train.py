"""Train one model and keep its best checkpoint (lowest validation loss).

Examples
--------
Full training run used for the report (writes models/best_<model>.npz):
    python train.py --model dae
    python train.py --model vae --beta 1.0
    python train.py --model gnn

Live demo during the evaluation (a few epochs, does NOT overwrite the
submitted checkpoints because it writes to models/demo/):
    python train.py --model gnn --epochs 3 --demo
    python train.py --model vae --epochs 2 --subset 1000 --demo
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import sys as _sys
from floorplan import config as _C
if "--labels" in _sys.argv:
    _C.set_labels(_sys.argv[_sys.argv.index("--labels") + 1])
from floorplan import config as C
from floorplan.data import load_splits, fixed_noisy_sets
from floorplan.models import build_model
from floorplan.trainer import fit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["dae", "vae", "gnn"])
    p.add_argument("--labels", default="ink", choices=["ink", "walls"], help="ink = dark pixels of the PNG (mid-review); walls = official SVG wall polygons")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=C.TRAIN_SEED)
    p.add_argument("--subset", type=int, default=0, help="train on the first N plans only (demo)")
    p.add_argument("--demo", action="store_true", help="write to models/demo/ instead of models/")
    p.add_argument("--tag", default="", help="suffix for the checkpoint / history file names")
    # model-specific
    p.add_argument("--width", type=int, default=16, help="DAE / VAE conv width")
    p.add_argument("--beta", type=float, default=C.VAE_BETA, help="VAE KL weight")
    p.add_argument("--kl-warmup", type=int, default=C.VAE_KL_WARMUP, help="VAE warm-up epochs")
    p.add_argument("--hidden", type=int, default=C.GNN_HIDDEN, help="GNN node embedding size")
    p.add_argument("--layers", type=int, default=C.GNN_LAYERS, help="GNN message-passing layers")
    return p.parse_args()


def main():
    args = parse_args()
    name = args.model.upper()
    kwargs = {"seed": args.seed}
    if name in ("DAE", "VAE"):
        kwargs["width"] = args.width
    if name == "VAE":
        kwargs["beta"] = args.beta
    if name == "GNN":
        kwargs.update(hidden=args.hidden, n_layers=args.layers)

    X_train, X_val, X_test = load_splits()
    if args.subset:
        X_train = X_train[:args.subset]
    val_noisy, _ = fixed_noisy_sets(X_val, X_test)

    model = build_model(name, **kwargs)
    out_dir = C.MODELS_DIR / "demo" if args.demo else C.MODELS_DIR
    stem = f"best_{args.model}{('_' + args.tag) if args.tag else ''}"
    ckpt = out_dir / f"{stem}.npz"
    print(f"Training {name} ({model.n_params():,} parameters) for {args.epochs} epochs "
          f"on {len(X_train)} plans; hyper-parameters {json.dumps(model.hparams())}")

    history = fit(model, X_train, X_val, val_noisy, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, seed=args.seed, ckpt_path=ckpt,
                  kl_warmup=args.kl_warmup if name == "VAE" else 0)

    res_dir = C.RESULTS_DIR / "demo" if args.demo else C.RESULTS_DIR
    os.makedirs(res_dir, exist_ok=True)
    table = pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list)})
    table.to_csv(res_dir / f"history_{stem.replace('best_', '')}.csv", index=False)
    print(f"Best checkpoint: {ckpt}  (epoch {history['best_epoch']}, "
          f"val loss {history['best_val_loss']:.4f})")


if __name__ == "__main__":
    main()
