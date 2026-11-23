"""Numerical gradient check for the hand-written backward passes of all models.

Central finite differences in float64, probed at two step sizes (h and h/2).
A coordinate whose two estimates disagree sits on a ReLU kink, where a finite
difference is not valid, so it is discarded rather than counted as a failure.
Agreement is judged with the numpy.allclose rule |num - ana| <= atol + rtol*|ana|.
"""
import numpy as np

from .models import build_model


def gradient_check(name, seed=0, n=4, n_coords=12, h=1e-6, kink_tol=1e-3,
                   atol=1e-5, rtol=1e-3, verbose=True, **model_kwargs):
    rng = np.random.default_rng(seed)
    model = build_model(name, seed=seed + 1, **model_kwargs).to_float64()
    # Move biases off zero: an all-background window gives z = b = 0 exactly,
    # i.e. a ReLU kink, at initialisation.
    for layer in model.param_layers():
        layer.b = layer.b + rng.standard_normal(layer.b.shape) * 0.05

    side2 = model.side * model.side
    x = (rng.random((n, side2)) > 0.8).astype(np.float64)
    y = (rng.random((n, side2)) > 0.85).astype(np.float64)
    noise_seed = 1234        # the VAE's epsilon must be identical in every call
    _, grads = model.loss_and_grads(x, y, np.random.default_rng(noise_seed))
    params = model.parameters()

    def loss_at(flat, j, delta):
        old = flat[j]
        flat[j] = old + delta
        value = model.loss(x, y, np.random.default_rng(noise_seed))
        flat[j] = old
        return value

    worst, n_kink, n_smooth = 0.0, 0, 0
    for k, (p, g) in enumerate(zip(params, grads)):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        idx = rng.choice(flat.size, size=min(n_coords, flat.size), replace=False)
        for j in idx:
            coarse = (loss_at(flat, j, h) - loss_at(flat, j, -h)) / (2 * h)
            fine = (loss_at(flat, j, h / 2) - loss_at(flat, j, -h / 2)) / h
            if abs(coarse - fine) / max(abs(coarse), abs(fine), 1e-12) > kink_tol:
                n_kink += 1
                continue
            n_smooth += 1
            worst = max(worst, abs(fine - gflat[j]) / (atol + rtol * abs(gflat[j])))
    ok = worst <= 1.0
    if verbose:
        print(f"  {name}: {len(params)} tensors, {n_smooth} smooth probes "
              f"({n_kink} on ReLU kinks discarded), worst tolerance ratio {worst:.3f} "
              f"-> {'CORRECT' if ok else 'WRONG'}")
    return ok, worst
