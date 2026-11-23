"""The three models compared in the final review, all written from scratch.

Every model exposes the same interface, so the trainer, the evaluation and the
gradient check never need to know which one they are handling:

    parameters()                 -> list of weight arrays (updated in place by Adam)
    predict(x_noisy)             -> wall probabilities, shape (N, 1024), deterministic
    loss_and_grads(x, y, rng, kl_scale) -> (dict of loss terms, list of gradients)
    loss(x, y, rng)              -> scalar total loss (forward only, gradient check)
    hparams()                    -> dict saved in the checkpoint

DAE  convolutional denoising autoencoder (the mid-review model, unchanged)
VAE  convolutional variational autoencoder with the same backbone: the encoder
     outputs a mean and a log-variance, so the decoder can also be sampled
     from the N(0, I) prior to GENERATE new plans
GNN  GraphSAGE network on a graph of 4x4-pixel patches (64 nodes, 8-neighbour
     edges); every node predicts its own 16 pixels from messages passed
     between neighbouring patches
"""
import numpy as np

from . import config as C
from .layers import Conv, Linear, ReLU, Upsample, sigmoid, run_forward, run_backward
from .losses import bce_loss, bce_logit_grad


class BaseModel:
    name = "base"
    wall_weight = C.WALL_WEIGHT

    def param_layers(self):
        """Layers that own weights, in a fixed order (defines parameters())."""
        raise NotImplementedError

    def parameters(self):
        out = []
        for layer in self.param_layers():
            out += [layer.W, layer.b]
        return out

    def n_params(self):
        return int(sum(p.size for p in self.parameters()))

    def set_parameters(self, arrays):
        arrays = list(arrays)
        for layer in self.param_layers():
            layer.W, layer.b = arrays.pop(0), arrays.pop(0)

    def to_float64(self):
        """Used by the gradient check: finite differences need double precision."""
        self.set_parameters([p.astype(np.float64) for p in self.parameters()])
        return self

    def _flat_grads(self, layer_list, grad_map):
        """Order the {layer index: (gW, gb)} map of run_backward like parameters()."""
        out = []
        for i, layer in enumerate(layer_list):
            if layer.has_params:
                out += list(grad_map[i])
        return out


def _decoder(c1, c2, rng):
    """Shared decoder: three (conv -> ReLU -> 2x upsample) stages, then a 1x1 conv."""
    return [
        Conv(c1, c2, 3, 1, 1, rng), ReLU(), Upsample(),    #  4 ->  8
        Conv(c2, c1, 3, 1, 1, rng), ReLU(), Upsample(),    #  8 -> 16
        Conv(c1, c1, 3, 1, 1, rng), ReLU(), Upsample(),    # 16 -> 32
        Conv(c1, 1, 1, 1, 0, rng),                         # logits; sigmoid applied outside
    ]


# =============================================================================
# 1. Denoising autoencoder
# =============================================================================
class DAE(BaseModel):
    """1x32x32 -> 16x16x16 -> 32x8x8 -> [16x4x4 latent] -> ... -> 1x32x32."""
    name = "DAE"

    def __init__(self, width=C.DAE_WIDTH, seed=C.TRAIN_SEED, side=C.SIDE):
        rng = np.random.default_rng(seed)
        c1, c2 = width, 2 * width
        self.width, self.side, self.seed = width, side, seed
        self.encoder = [Conv(1, c1, 3, 2, 1, rng), ReLU(),
                        Conv(c1, c2, 3, 2, 1, rng), ReLU(),
                        Conv(c2, c1, 3, 2, 1, rng), ReLU()]      # latent is non-negative
        self.decoder = _decoder(c1, c2, rng)

    def param_layers(self):
        return [l for l in self.encoder + self.decoder if l.has_params]

    def hparams(self):
        return {"model": self.name, "width": self.width, "seed": self.seed}

    def encode(self, x):
        return run_forward(self.encoder, x.reshape(x.shape[0], 1, self.side, self.side))

    def decode(self, z):
        logits = run_forward(self.decoder, z)
        return sigmoid(logits).reshape(z.shape[0], -1)

    def predict(self, x):
        return self.decode(self.encode(x))

    def loss(self, x, y, rng=None):
        return bce_loss(self.predict(x), y, self.wall_weight)

    def loss_and_grads(self, x, y, rng=None, kl_scale=1.0):
        prob = self.predict(x)
        d = bce_logit_grad(prob, y, self.wall_weight).reshape(x.shape[0], 1, self.side, self.side)
        d, g_dec = run_backward(self.decoder, d)
        _, g_enc = run_backward(self.encoder, d)
        grads = self._flat_grads(self.encoder, g_enc) + self._flat_grads(self.decoder, g_dec)
        recon = bce_loss(prob, y, self.wall_weight)
        return {"loss": recon, "recon": recon}, grads


# =============================================================================
# 2. Variational autoencoder
# =============================================================================
class VAE(BaseModel):
    """Same backbone as the DAE, but the last encoder stage is split into two
    heads that output mu and log sigma^2 of q(z|x) = N(mu, diag(sigma^2)).

        z = mu + sigma * eps,  eps ~ N(0, I)          (reparameterisation trick)
        L = BCE_w(x_hat, x) + beta * KL(q(z|x) || N(0, I)) / D
        KL = -1/2 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)

    D = 1024 pixels, because the BCE term is a per-pixel mean: dividing KL by D
    keeps the two terms on the same scale, so beta = 1 is the usual ELBO.
    """
    name = "VAE"

    def __init__(self, width=C.VAE_WIDTH, beta=C.VAE_BETA, seed=C.TRAIN_SEED, side=C.SIDE):
        rng = np.random.default_rng(seed)
        c1, c2 = width, 2 * width
        self.width, self.beta, self.side, self.seed = width, beta, side, seed
        self.encoder = [Conv(1, c1, 3, 2, 1, rng), ReLU(),
                        Conv(c1, c2, 3, 2, 1, rng), ReLU()]
        self.mu_head = Conv(c2, c1, 3, 2, 1, rng)
        self.lv_head = Conv(c2, c1, 3, 2, 1, rng)
        self.decoder = _decoder(c1, c2, rng)
        self.latent_shape = (c1, side // 8, side // 8)

    def param_layers(self):
        return ([l for l in self.encoder if l.has_params] + [self.mu_head, self.lv_head]
                + [l for l in self.decoder if l.has_params])

    def hparams(self):
        return {"model": self.name, "width": self.width, "beta": self.beta, "seed": self.seed}

    def encode(self, x):
        h = run_forward(self.encoder, x.reshape(x.shape[0], 1, self.side, self.side))
        return self.mu_head.forward(h), self.lv_head.forward(h)

    def decode(self, z):
        return sigmoid(run_forward(self.decoder, z)).reshape(z.shape[0], -1)

    def predict(self, x):
        """Repair uses the posterior mean, so predictions are deterministic."""
        mu, _ = self.encode(x)
        return self.decode(mu)

    def sample_prior(self, n, rng):
        """Generate brand-new plans: z ~ N(0, I), then decode."""
        return self.decode(rng.standard_normal((n,) + self.latent_shape).astype(np.float32))

    def _forward_train(self, x, rng):
        mu, lv = self.encode(x)
        sigma = np.exp(0.5 * np.clip(lv, -20, 20))
        eps = rng.standard_normal(mu.shape).astype(mu.dtype)
        z = mu + sigma * eps
        prob = self.decode(z)
        return mu, lv, sigma, eps, prob

    def _kl(self, mu, lv):
        """Mean over the batch of the per-sample KL divergence (in nats)."""
        per = -0.5 * (1 + lv - mu ** 2 - np.exp(np.clip(lv, -20, 20))).reshape(mu.shape[0], -1).sum(1)
        return float(per.mean())

    def loss(self, x, y, rng, kl_scale=1.0):
        mu, lv, _, _, prob = self._forward_train(x, rng)
        D = self.side * self.side
        return bce_loss(prob, y, self.wall_weight) + kl_scale * self.beta * self._kl(mu, lv) / D

    def loss_and_grads(self, x, y, rng, kl_scale=1.0):
        n, D = x.shape[0], self.side * self.side
        mu, lv, sigma, eps, prob = self._forward_train(x, rng)
        # reconstruction term -> decoder -> dL/dz
        d = bce_logit_grad(prob, y, self.wall_weight).reshape(n, 1, self.side, self.side)
        dz, g_dec = run_backward(self.decoder, d)
        # reparameterisation + KL:  c = beta / (N * D)
        c = kl_scale * self.beta / (n * D)
        dmu = dz + c * mu
        dlv = dz * eps * 0.5 * sigma + c * 0.5 * (sigma ** 2 - 1.0)
        dh_mu, gW_mu, gb_mu = self.mu_head.backward(dmu)
        dh_lv, gW_lv, gb_lv = self.lv_head.backward(dlv)
        _, g_enc = run_backward(self.encoder, dh_mu + dh_lv)
        grads = (self._flat_grads(self.encoder, g_enc) + [gW_mu, gb_mu, gW_lv, gb_lv]
                 + self._flat_grads(self.decoder, g_dec))
        recon = bce_loss(prob, y, self.wall_weight)
        kl = self._kl(mu, lv)
        return {"loss": recon + kl_scale * self.beta * kl / D, "recon": recon, "kl": kl}, grads


# =============================================================================
# 3. Graph neural network (GraphSAGE on a patch graph)
# =============================================================================
def grid_adjacency(g):
    """Row-normalised 8-neighbour adjacency of a g x g grid (no self-loops).

    Mean aggregation: (A h)_u = (1 / |N(u)|) * sum_{v in N(u)} h_v
    """
    A = np.zeros((g * g, g * g), np.float32)
    for r in range(g):
        for c in range(g):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < g and 0 <= cc < g:
                        A[r * g + c, rr * g + cc] = 1.0
    return A / A.sum(axis=1, keepdims=True)


class SAGELayer:
    """Residual GraphSAGE layer with a mean aggregator:

        h_u' = h_u + ReLU( W_self h_u + W_neigh * mean_{v in N(u)} h_v + b )
    """
    has_params = True

    def __init__(self, dim, A, rng):
        self.A = A
        std = np.sqrt(2.0 / (2 * dim))
        # Stack W_self and W_neigh into one (2*dim, dim) matrix -> one matmul
        self.W = (rng.standard_normal((2 * dim, dim)) * std).astype(np.float32)
        self.b = np.zeros(dim, np.float32)

    def forward(self, h):
        self.agg = self.A @ h                                  # (N, V, F) mean of neighbours
        self.cat = np.concatenate([h, self.agg], axis=-1)      # [h_u || mean h_v]
        z = self.cat @ self.W + self.b
        self.mask = z > 0
        return h + np.maximum(z, 0.0)

    def backward(self, d):
        dz = d * self.mask
        F = d.shape[-1]
        gW = self.cat.reshape(-1, 2 * F).T @ dz.reshape(-1, F)
        gb = dz.reshape(-1, F).sum(axis=0)
        dcat = dz @ self.W.T
        dh = d + dcat[..., :F] + self.A.T @ dcat[..., F:]
        return dh, gW, gb


class GNN(BaseModel):
    """Image -> graph of P x P patches -> GraphSAGE -> per-node pixel logits.

    Node features: the P*P (damaged) pixels of the patch plus its normalised
    (row, col) position. Message passing is permutation-invariant, so without the
    position the network could not tell a patch at the edge of the plan from one
    in the middle.
    """
    name = "GNN"

    def __init__(self, hidden=C.GNN_HIDDEN, n_layers=C.GNN_LAYERS, patch=C.GNN_PATCH,
                 seed=C.TRAIN_SEED, side=C.SIDE):
        rng = np.random.default_rng(seed)
        self.hidden, self.n_layers, self.patch, self.seed, self.side = hidden, n_layers, patch, seed, side
        self.g = side // patch                                 # 8 x 8 = 64 nodes
        self.A = grid_adjacency(self.g)
        rows, cols = np.divmod(np.arange(self.g * self.g), self.g)
        self.pos = (np.stack([rows, cols], 1) / (self.g - 1) * 2 - 1).astype(np.float32)
        self.inp = Linear(patch * patch + 2, hidden, rng)
        self.relu = ReLU()
        self.sage = [SAGELayer(hidden, self.A, rng) for _ in range(n_layers)]
        self.out = Linear(hidden, patch * patch, rng)

    def param_layers(self):
        return [self.inp] + self.sage + [self.out]

    def hparams(self):
        return {"model": self.name, "hidden": self.hidden, "n_layers": self.n_layers,
                "patch": self.patch, "seed": self.seed}

    # image (N, 1024) <-> node features (N, V, P*P)
    def to_nodes(self, x):
        n, g, p = x.shape[0], self.g, self.patch
        return x.reshape(n, g, p, g, p).transpose(0, 1, 3, 2, 4).reshape(n, g * g, p * p)

    def to_image(self, nodes):
        n, g, p = nodes.shape[0], self.g, self.patch
        return nodes.reshape(n, g, g, p, p).transpose(0, 1, 3, 2, 4).reshape(n, -1)

    def _logits(self, x):
        return self.out.forward(self.node_embeddings(x))

    def node_embeddings(self, x):
        """Final hidden state of every node, shape (N, 64, hidden)."""
        nodes = self.to_nodes(x)
        pos = np.broadcast_to(self.pos, (x.shape[0],) + self.pos.shape).astype(nodes.dtype)
        h = self.relu.forward(self.inp.forward(np.concatenate([nodes, pos], -1)))
        for layer in self.sage:
            h = layer.forward(h)
        return h

    def predict(self, x):
        return self.to_image(sigmoid(self._logits(x)))

    def loss(self, x, y, rng=None):
        return bce_loss(self.predict(x), y, self.wall_weight)

    def loss_and_grads(self, x, y, rng=None, kl_scale=1.0):
        prob = self.predict(x)
        d = self.to_nodes(bce_logit_grad(prob, y, self.wall_weight))
        d, gW_out, gb_out = self.out.backward(d)
        sage_grads = []
        for layer in reversed(self.sage):
            d, gW, gb = layer.backward(d)
            sage_grads = [gW, gb] + sage_grads
        d = self.relu.backward(d)
        _, gW_in, gb_in = self.inp.backward(d)
        grads = [gW_in, gb_in] + sage_grads + [gW_out, gb_out]
        recon = bce_loss(prob, y, self.wall_weight)
        return {"loss": recon, "recon": recon}, grads


MODEL_CLASSES = {"DAE": DAE, "VAE": VAE, "GNN": GNN}


def build_model(name, **kwargs):
    return MODEL_CLASSES[name.upper()](**kwargs)
