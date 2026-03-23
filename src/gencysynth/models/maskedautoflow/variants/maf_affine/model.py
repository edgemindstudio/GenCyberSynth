# src/gencysynth/models/maskedautoflow/variants/maf_affine/model.py
# =============================================================================
# GenCyberSynth — MaskedAutoFlow — maf_affine — Model (Rule A compatible)
#
# This file defines the *model components only* (no filesystem I/O).
# Artifact reading/writing is handled by train.py / sample.py / pipeline.py.
#
# Highlights
# ----------
# • MADE_style masked dense layers with masks stored as non_trainable weights.
# • Reproducible degree assignments and per_flow permutations via RANDOM_STATE.
# • Keras 3–compatible (weights saved/loaded via model.save_weights/load_weights).
#
# API surface
# -----------
# - MAFConfig: config dataclass
# - MaskedDense, MADE, MAF: core components
# - build_maf_model: factory to create a built MAF (variables materialized)
# - flatten_images / reshape_to_images: light helpers (model_agnostic)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import numpy as np
import tensorflow as tf


# =============================================================================
# Config
# =============================================================================

@dataclass
class MAFConfig:
    """
    Configuration for a Masked Autoregressive Flow (MAF).

    Notes
    -----
    - IMG_SHAPE is used by helper utilities. The flow itself always operates on
      flattened vectors of size D = H*W*C.
    - RANDOM_STATE seeds both:
        (1) the NumPy RNG used for degree assignments and permutations, and
        (2) TensorFlow's RNG when callers set it (train/pipeline do that).
    """
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)

    # Training_time knobs (kept here for convenience; training lives elsewhere)
    LR: float = 2e_4
    CLIP_GRAD: float = 1.0
    PATIENCE: int = 10

    # Deterministic masks/permutations when set
    RANDOM_STATE: Optional[int] = 42


# =============================================================================
# Small helpers (shape transforms)
# =============================================================================

def flatten_images(
    x: np.ndarray,
    img_shape: Tuple[int, int, int],
    assume_01: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert images to flat vectors suitable for density estimation.

    Accepts
    -------
    x : np.ndarray
        Either (N, H, W, C) or already_flattened (N, D).

    Returns
    -------
    x_flat : np.ndarray
        (N, D) float32 in [0,1] (optionally clipped).

    Notes
    -----
    - If `assume_01` is True (default), we assume values are already in [0,1].
      If values look byte_like (max > 1.5), we scale by 255 as a best_effort.
    """
    x = np.asarray(x)
    if x.ndim == 4:
        H, W, C = img_shape
        x = x.reshape((-1, H * W * C))

    x = x.astype(np.float32, copy=False)

    # Best_effort normalization for legacy byte arrays
    if assume_01 and float(np.nanmax(x)) > 1.5:
        x = x / 255.0

    if clip:
        x = np.clip(x, 0.0, 1.0)

    return x


def reshape_to_images(
    x_flat: np.ndarray,
    img_shape: Tuple[int, int, int],
    clip: bool = True,
) -> np.ndarray:
    """
    Convert flat vectors back to images.

    Returns
    -------
    imgs : (N, H, W, C) float32 in [0,1]
    """
    H, W, C = img_shape
    x = np.asarray(x_flat, dtype=np.float32)
    x = x.reshape((-1, H, W, C))
    if clip:
        x = np.clip(x, 0.0, 1.0)
    return x


# =============================================================================
# Core masked layers / MADE / MAF
# =============================================================================

class MaskedDense(tf.keras.layers.Layer):
    """
    Dense layer with a fixed binary mask applied to the kernel to enforce
    autoregressive constraints (MADE_style).

    Why store masks as non_trainable weights?
    ----------------------------------------
    Keeping masks as raw tensors created inside build/call can cause graph
    capture/scoping issues under autograph/`@tf.function`. Storing as a
    `trainable=False` weight makes it graph_safe and checkpoint_friendly.
    """

    def __init__(
        self,
        units: int,
        in_degrees: np.ndarray,
        out_degrees: np.ndarray,
        *,
        use_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.units = int(units)

        # Degrees are small, immutable NumPy arrays (safe to store in Python)
        self.in_degrees_np = np.asarray(in_degrees, dtype=np.int32)
        self.out_degrees_np = np.asarray(out_degrees, dtype=np.int32)
        self.use_bias = bool(use_bias)

        # Created in build()
        self.kernel: tf.Variable
        self.bias: Optional[tf.Variable]
        self.mask: tf.Variable  # non_trainable

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != int(self.in_degrees_np.shape[0]):
            raise ValueError(
                f"MaskedDense expected input dim {self.in_degrees_np.shape[0]}, got {in_dim}"
            )

        # Trainable parameters
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        ) if self.use_bias else None

        # Mask rule: connection allowed if in_degree <= out_degree
        mask_np = (self.in_degrees_np[:, None] <= self.out_degrees_np[None, :]).astype("float32")

        # Store mask as a non_trainable weight so it lives in the right graph
        self.mask = self.add_weight(
            name="mask",
            shape=mask_np.shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(mask_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        masked_kernel = tf.multiply(self.kernel, self.mask)
        y = tf.linalg.matmul(inputs, masked_kernel)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        return y


class MADE(tf.keras.Model):
    """
    MADE network: masked MLP producing per_dimension (mu, log_sigma).

    Degree construction is reproducible using the provided `seed`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        rng = np.random.default_rng(seed)

        # Input degrees are fixed in ascending order
        deg_in = np.arange(1, self.input_dim + 1, dtype=np.int32)

        # Hidden degrees are random in [1, D]
        deg_hidden = [
            rng.integers(1, self.input_dim + 1, size=h, dtype=np.int32)
            for h in self.hidden_dims
        ]

        # Output degrees match the input ordering
        deg_out = np.arange(1, self.input_dim + 1, dtype=np.int32)

        # Masked hidden stack
        layers_seq: List[tf.keras.layers.Layer] = []
        prev_deg = deg_in
        for li, h in enumerate(self.hidden_dims):
            layers_seq.append(
                MaskedDense(h, in_degrees=prev_deg, out_degrees=deg_hidden[li], name=f"mdense_{li}")
            )
            layers_seq.append(tf.keras.layers.ReLU())
            prev_deg = deg_hidden[li]

        self.net = layers_seq

        # Separate heads for mu and log_sigma
        self.mu_layer = MaskedDense(self.input_dim, in_degrees=prev_deg, out_degrees=deg_out, name="mu")
        self.log_sigma_layer = MaskedDense(self.input_dim, in_degrees=prev_deg, out_degrees=deg_out, name="log_sigma")

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h = x
        for layer in self.net:
            h = layer(h)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)

        # Clamp for numerical stability (prevents extreme exp/logdet)
        log_sigma = tf.clip_by_value(log_sigma, -7.0, 7.0)
        return mu, log_sigma


class MAF(tf.keras.Model):
    """
    Masked Autoregressive Flow (stack of MADE transforms).

    Forward (density):
      x -> permute -> affine autoregressive transform -> z

    Inverse (sampling):
      z -> invert affine autoregressive (sequential over dims) -> inverse permute -> x
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_flows: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
        random_state: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.num_flows = int(num_flows)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        base = 0 if random_state is None else int(random_state)
        rng = np.random.default_rng(base)

        # Deterministic permutations per flow
        self._perms_np: List[np.ndarray] = [
            rng.permutation(self.input_dim).astype(np.int32) for _ in range(self.num_flows)
        ]
        self._inv_perms_np: List[np.ndarray] = [np.argsort(p).astype(np.int32) for p in self._perms_np]

        # TF tensors created in build()
        self._perms_tf: List[tf.Tensor] = []
        self._inv_perms_tf: List[tf.Tensor] = []

        # MADE blocks (each gets its own deterministic seed)
        self.flows: List[MADE] = [
            MADE(self.input_dim, self.hidden_dims, seed=base + i, name=f"made_{i}")
            for i in range(self.num_flows)
        ]

    def build(self, input_shape):
        # Permutations must be constants in the current graph/context
        self._perms_tf = [tf.constant(p, dtype=tf.int32) for p in self._perms_np]
        self._inv_perms_tf = [tf.constant(p, dtype=tf.int32) for p in self._inv_perms_np]

        # Warm_build all variables by running once
        dummy = tf.zeros((1, self.input_dim), dtype=tf.float32)
        _ = self.call(dummy)
        super().build(input_shape)

    # -------------------------------------------------------------------------
    # Forward transform and log_det
    # -------------------------------------------------------------------------
    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply forward transform x -> z.

        Returns
        -------
        z : Tensor (B, D)
        log_det : Tensor (B,)     log|det(dz/dx)|
        """
        z = x
        log_det = tf.zeros((tf.shape(x)[0],), dtype=tf.float32)

        for i, flow in enumerate(self.flows):
            # Permute before each flow
            z = tf.gather(z, self._perms_tf[i], axis=1)

            # Autoregressive affine parameters
            mu, log_sigma = flow(z)

            # Forward affine: z <- (z - mu) * exp(-log_sigma)
            z = (z - mu) * tf.exp(-log_sigma)

            # Log_det for diagonal affine transform
            log_det += -tf.reduce_sum(log_sigma, axis=1)

        return z, log_det

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Keras_friendly call: returns z only."""
        z, _ = self.forward(x)
        return z

    # -------------------------------------------------------------------------
    # Log_likelihood under standard normal base
    # -------------------------------------------------------------------------
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(x) for a standard normal base distribution.
        """
        z, log_det = self.forward(x)
        log_base = -0.5 * tf.reduce_sum(z * z + tf.math.log(2.0 * np.pi), axis=1)
        return log_base + log_det

    # -------------------------------------------------------------------------
    # Inverse (sampling)
    # -------------------------------------------------------------------------
    @staticmethod
    def _inverse_single_flow(z: tf.Tensor, flow: MADE, input_dim: int) -> tf.Tensor:
        """
        Invert one autoregressive affine flow.

        For each dimension i, solve:
          z_i = (x_i - mu_i(x_<i)) * exp(-log_sigma_i(x_<i))
        => x_i = mu_i(x_<i) + exp(log_sigma_i(x_<i)) * z_i

        Because mu/log_sigma for dim i depend only on x_<i (via masks),
        we can recover x sequentially from i=0..D_1.
        """
        batch = tf.shape(z)[0]
        D = int(input_dim)  # static for our use
        x = tf.zeros_like(z)

        for i in range(D):
            mu, log_sigma = flow(x)
            xi = mu[:, i] + tf.exp(log_sigma[:, i]) * z[:, i]

            xi = tf.reshape(xi, (batch, 1))
            left = x[:, :i]
            right = x[:, i + 1 :]
            x = tf.concat([left, xi, right], axis=1)

        return x

    def inverse(self, z: tf.Tensor) -> tf.Tensor:
        """
        Invert the full flow stack (reverse order).

        For each flow in reverse:
          1) invert affine autoregressive transform
          2) invert permutation
        """
        x = z
        for i, flow in reversed(list(enumerate(self.flows))):
            x = self._inverse_single_flow(x, flow, self.input_dim)
            x = tf.gather(x, self._inv_perms_tf[i], axis=1)
        return x


# =============================================================================
# Factory
# =============================================================================

def build_maf_model(
    cfg_or_dim: Union[int, MAFConfig],
    num_layers: Optional[int] = None,
    hidden_dims: Optional[Iterable[int]] = None,
) -> MAF:
    """
    Build a MAF instance from either:
      - a MAFConfig, or
      - a bare dimension + optional overrides.

    Returns
    -------
    maf : MAF
        A *built* model (variables created), ready for save/load/training.

    Examples
    --------
    >>> maf = build_maf_model(1600, num_layers=5, hidden_dims=(128,128))
    >>> maf = build_maf_model(MAFConfig(IMG_SHAPE=(40,40,1), NUM_FLOWS=6))
    """
    if isinstance(cfg_or_dim, MAFConfig):
        D = int(np.prod(cfg_or_dim.IMG_SHAPE))
        flows = int(cfg_or_dim.NUM_FLOWS)
        h = tuple(int(v) for v in cfg_or_dim.HIDDEN_DIMS)
        maf = MAF(
            input_dim=D,
            num_flows=flows,
            hidden_dims=h,
            random_state=cfg_or_dim.RANDOM_STATE,
            name="MAF",
        )
    else:
        D = int(cfg_or_dim)
        flows = int(num_layers) if num_layers is not None else 5
        h = tuple(int(v) for v in (hidden_dims if hidden_dims is not None else (128, 128)))
        maf = MAF(
            input_dim=D,
            num_flows=flows,
            hidden_dims=h,
            random_state=None,
            name="MAF",
        )

    # Eager_build variables + permutation tensors (safe for checkpoints)
    maf.build((None, D))
    return maf


__all__ = [
    "MAFConfig",
    "MaskedDense",
    "MADE",
    "MAF",
    "build_maf_model",
    "flatten_images",
    "reshape_to_images",
]