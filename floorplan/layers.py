"""Hand-written neural-network building blocks (NumPy only).

Each layer caches what its backward pass needs during forward(). Layers with
weights return (d_input, grad_W, grad_b) from backward(); the others return
d_input only.
"""
import numpy as np


# ---- convolution via im2col ----------------------------------------------------
def im2col(x, k, stride, pad):
    """(N, C, H, W) -> (N, out_h*out_w, C*k*k): every sliding window as a row."""
    n, c, h, w = x.shape
    out_h = (h + 2 * pad - k) // stride + 1
    out_w = (w + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    s = xp.strides
    windows = np.lib.stride_tricks.as_strided(
        xp, shape=(n, c, k, k, out_h, out_w),
        strides=(s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride),
        writeable=False)
    cols = windows.transpose(0, 4, 5, 1, 2, 3).reshape(n, out_h * out_w, c * k * k)
    return np.ascontiguousarray(cols), out_h, out_w


def col2im(cols, shape, k, stride, pad):
    """Transpose of im2col: scatter patches back and ACCUMULATE the overlaps."""
    n, c, h, w = shape
    out_h = (h + 2 * pad - k) // stride + 1
    out_w = (w + 2 * pad - k) // stride + 1
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    view = cols.reshape(n, out_h, out_w, c * k * k).transpose(0, 3, 1, 2)
    for ki in range(k):
        for kj in range(k):
            xp[:, :, ki:ki + stride * out_h:stride,
               kj:kj + stride * out_w:stride] += view[:, ki * k + kj::k * k, :, :]
    return xp[:, :, pad:pad + h, pad:pad + w]


class Conv:
    """2-D convolution = im2col + one matrix multiply. Xavier initialisation."""
    has_params = True

    def __init__(self, c_in, c_out, k, stride, pad, rng):
        self.c_in, self.c_out, self.k, self.stride, self.pad = c_in, c_out, k, stride, pad
        fan_in, fan_out = c_in * k * k, c_out * k * k
        self.W = (rng.standard_normal((c_out, c_in, k, k))
                  * np.sqrt(2.0 / (fan_in + fan_out))).astype(np.float32)
        self.b = np.zeros(c_out, np.float32)

    def forward(self, x):
        self.cols, oh, ow = im2col(x, self.k, self.stride, self.pad)
        self.x_shape = x.shape
        z = self.cols.reshape(-1, self.c_in * self.k ** 2) @ self.W.reshape(self.c_out, -1).T + self.b
        return z.reshape(x.shape[0], oh, ow, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, d):
        dout = d.transpose(0, 2, 3, 1).reshape(-1, self.c_out)
        wmat = self.W.reshape(self.c_out, -1)
        gW = (dout.T @ self.cols.reshape(-1, self.c_in * self.k ** 2)).reshape(self.W.shape)
        gb = dout.sum(axis=0)
        dcols = (dout @ wmat).reshape(self.cols.shape)
        return col2im(dcols, self.x_shape, self.k, self.stride, self.pad), gW, gb


class Linear:
    """Dense layer applied to the last axis of an array of any rank."""
    has_params = True

    def __init__(self, f_in, f_out, rng):
        self.W = (rng.standard_normal((f_in, f_out))
                  * np.sqrt(2.0 / (f_in + f_out))).astype(np.float32)
        self.b = np.zeros(f_out, np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = d.reshape(-1, d.shape[-1])
        return d @ self.W.T, x2.T @ d2, d2.sum(axis=0)


class ReLU:
    has_params = False

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, d):
        return d * self.mask


class Upsample:
    """Nearest-neighbour 2x upsampling; the gradient sums each 2x2 block."""
    has_params = False

    def forward(self, x):
        return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

    def backward(self, d):
        h, w = d.shape[2], d.shape[3]
        return (d[:, :, 0:h:2, 0:w:2] + d[:, :, 1:h:2, 0:w:2]
                + d[:, :, 0:h:2, 1:w:2] + d[:, :, 1:h:2, 1:w:2])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_backward(layers, d):
    """Reverse sweep through a list of layers. Returns (d_input, {idx: (gW, gb)})."""
    grads = {}
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        if layer.has_params:
            d, gW, gb = layer.backward(d)
            grads[i] = (gW, gb)
        else:
            d = layer.backward(d)
    return d, grads


def run_forward(layers, x):
    for layer in layers:
        x = layer.forward(x)
    return x


# ---- optimiser -----------------------------------------------------------------
class Adam:
    """theta <- theta - lr * m_hat / (sqrt(v_hat) + eps), updated in place."""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for k, (p, g) in enumerate(zip(self.params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
