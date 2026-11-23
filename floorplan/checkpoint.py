"""Save and restore model weights together with the metadata needed to rebuild them."""
import json

import numpy as np

from .models import build_model


def save_checkpoint(model, path, **meta):
    """Write every weight array plus a JSON header (hyper-parameters, epoch, ...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"p{i}": p for i, p in enumerate(model.parameters())}
    header = dict(model.hparams(), **meta)
    payload["meta"] = np.array(json.dumps(header))
    np.savez(path, **payload)


def load_checkpoint(path):
    """Rebuild the model from a checkpoint. Returns (model, metadata dict)."""
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    kwargs = {k: v for k, v in meta.items()
              if k in ("width", "beta", "seed", "hidden", "n_layers", "patch")}
    model = build_model(meta["model"], **kwargs)
    n = len(model.parameters())
    model.set_parameters([data[f"p{i}"] for i in range(n)])
    return model, meta
