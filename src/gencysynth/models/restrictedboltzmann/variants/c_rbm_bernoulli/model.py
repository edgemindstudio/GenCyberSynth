# src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/model.py
"""
Restricted Boltzmann Machine (RBM) model components.

Rule A note (Artifacts)
-----------------------
This file is **model_only**. It MUST NOT read/write artifacts directly.
All checkpointing / synthetic writes / previews live in:
  - pipeline.py  (train + synth; Rule A artifact layout)
  - sample.py    (PNG/manifest synthesis helper; Rule A artifact layout)

Keeping model code artifact_free makes it reusable across:
  • multiple datasets
  • multiple orchestration paths (app/main, notebooks, unit tests)
  • multiple artifact layouts (if we ever migrate again)

What you get
------------
- RBMConfig:  dataclass of hyperparameters used by builders/pipelines.
- RBM:        Bernoulli–Bernoulli RBM (visible/hidden are Bernoulli).
- build_rbm:   convenience constructor (seeding + eager build for load_weights).
- Image helpers:
    * to_float01       : normalize to [0,1]
    * binarize01       : binarize in {0,1}
    * flatten_images   : (N,H,W,C) -> (N,V)
    * reshape_to_images: (N,V) -> (N,H,W,C)
- BernoulliRBM: wrapper with sampling methods expected by samplers:
    * sample_h_given_v
    * sample_v_given_h

Conventions
-----------
- Images are channels_last (H, W, C).
- Visible vectors are float32; for Bernoulli RBM training we often binarize.
- In CD mode, parameter updates are applied manually (classic RBM).
- In MSE mode, we treat the RBM as a reconstruction network and use autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Config
# =============================================================================
@dataclass
class RBMConfig:
    """
    Configuration for a Bernoulli–Bernoulli RBM.

    Notes
    -----
    - `visible_units` must match the flattened image dimension V = H*W*C.
    - `train_mode`:
        * "cd"  : contrastive divergence updates (classic RBM)
        * "mse" : reconstruction MSE loss with optimizer/autograd
    - `seed` is used by build_rbm() to seed TF/NumPy (best_effort reproducibility).
    """
    visible_units: int
    hidden_units: int = 256
    cd_k: int = 1
    learning_rate: float = 1e_3
    weight_decay: float = 0.0
    train_mode: str = "cd"  # {"cd", "mse"}
    seed: Optional[int] = 42


# =============================================================================
# Image helpers (shape + normalization)
# =============================================================================
def to_float01(x: np.ndarray) -> np.ndarray:
    """
    Convert an image array to float32 in [0,1].

    Accepts either:
      - float arrays already in [0,1]
      - uint8/float arrays in [0,255] (auto_normalized)
    """
    x = np.asarray(x).astype("float32", copy=False)
    if x.size > 0 and x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def binarize01(x: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Binarize values in [0,1] to {0,1} float32.
    """
    return (np.asarray(x) >= float(thresh)).astype("float32")


def flatten_images(
    x: np.ndarray,
    img_shape: Tuple[int, int, int],
    *,
    assume_01: bool = True,
) -> np.ndarray:
    """
    Flatten images to visible vectors.

    Parameters
    ----------
    x : np.ndarray
        (N,H,W,C) or already_flattened (N,V).
    img_shape : (H,W,C)
        Target image shape used if x is 4D.
    assume_01 : bool
        If False, auto_normalize from 0..255 via to_float01.

    Returns
    -------
    (N,V) float32 array.
    """
    x = np.asarray(x)
    H, W, C = img_shape

    if x.ndim == 4:
        x = x.reshape((-1, H, W, C))
        if not assume_01:
            x = to_float01(x)
        return x.reshape((-1, H * W * C)).astype("float32", copy=False)

    if x.ndim == 2:
        # Already flattened (N,V) — best effort normalization if requested.
        x = x.astype("float32", copy=False)
        if not assume_01 and x.size > 0 and x.max() > 1.5:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    raise ValueError(f"Expected x to be (N,H,W,C) or (N,V); got shape {x.shape}")


def reshape_to_images(x_flat: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Reshape visible vectors (N,V) back to images (N,H,W,C).
    """
    H, W, C = img_shape
    x = np.asarray(x_flat).astype("float32", copy=False)
    return x.reshape((-1, H, W, C))


# =============================================================================
# Core RBM
# =============================================================================
class RBM(tf.keras.Model):
    """
    Bernoulli–Bernoulli Restricted Boltzmann Machine.

    Parameters
    ----------
    visible_units : int
        Number of visible units V (flattened image dimension).
    hidden_units : int
        Number of hidden units H.

    Variables
    ---------
    W      : (V,H)
    h_bias : (H,)
    v_bias : (V,)

    Modes
    -----
    - CD (contrastive divergence): manual parameter updates.
    - MSE (reconstruction): autograd + optimizer.
    """

    def __init__(self, visible_units: int, hidden_units: int = 256, name: str = "rbm"):
        super().__init__(name=name)
        self.visible_units = int(visible_units)
        self.hidden_units = int(hidden_units)

        # Small normal init is standard for RBMs.
        init = tf.keras.initializers.RandomNormal(stddev=0.01)

        # NOTE: use tf.Variable directly (instead of add_weight) to keep this
        # simple and explicit; Keras save_weights/load_weights works fine.
        self.W = tf.Variable(init(shape=(self.visible_units, self.hidden_units)), name="W")
        self.h_bias = tf.Variable(tf.zeros([self.hidden_units], dtype=tf.float32), name="h_bias")
        self.v_bias = tf.Variable(tf.zeros([self.visible_units], dtype=tf.float32), name="v_bias")

    # ------------------------- Core ops -------------------------
    @tf.function(jit_compile=False)
    def _sigmoid(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.sigmoid(x)

    @tf.function(jit_compile=False)
    def _bernoulli_sample(self, probs: tf.Tensor) -> tf.Tensor:
        rnd = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)
        return tf.cast(rnd < probs, probs.dtype)

    @tf.function(jit_compile=False)
    def propup(self, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute hidden logits/probs given visible state.
        """
        logits = tf.linalg.matmul(v, self.W) + self.h_bias
        probs = self._sigmoid(logits)
        return logits, probs

    @tf.function(jit_compile=False)
    def propdown(self, h: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute visible logits/probs given hidden state.
        """
        logits = tf.linalg.matmul(h, tf.transpose(self.W)) + self.v_bias
        probs = self._sigmoid(logits)
        return logits, probs

    @tf.function(jit_compile=False)
    def gibbs_k(self, v0: tf.Tensor, k: int = 1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run k steps of blocked Gibbs sampling starting from v0.

        Returns
        -------
        v_k     : sampled visible state after k steps
        h_prob  : hidden probabilities at final step
        v_prob  : visible probabilities at final step
        """
        v = v0
        _, h_prob = self.propup(v)
        for _ in tf.range(int(k)):
            h = self._bernoulli_sample(h_prob)
            _, v_prob = self.propdown(h)
            v = self._bernoulli_sample(v_prob)
            _, h_prob = self.propup(v)
        return v, h_prob, v_prob

    # ------------------------- Forward recon (MSE mode) -------------------------
    @tf.function(jit_compile=False)
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        One_step reconstruction probabilities v -> h_prob -> v_prob.
        Used by MSE training and for validation metrics.
        """
        _, h_prob = self.propup(inputs)
        _, v_prob = self.propdown(h_prob)
        return v_prob

    # ------------------------- Free energy -------------------------
    @tf.function(jit_compile=False)
    def free_energy(self, v: tf.Tensor) -> tf.Tensor:
        """
        Free energy F(v) for Bernoulli–Bernoulli RBM.

        F(v) = -v·b - sum_j softplus((W^T v)_j + c_j)
        """
        vbias_term = tf.reduce_sum(v * self.v_bias, axis=1)
        hidden_lin = tf.linalg.matmul(v, self.W) + self.h_bias
        hidden_term = tf.reduce_sum(tf.math.softplus(hidden_lin), axis=1)
        return -(vbias_term + hidden_term)

    # ------------------------- Training steps -------------------------
    @tf.function(jit_compile=False)
    def train_step_cd(
        self,
        v0: tf.Tensor,
        *,
        k: int = 1,
        lr: float = 1e_3,
        weight_decay: float = 0.0,
    ) -> tf.Tensor:
        """
        One contrastive_divergence update (CD_k).

        Returns
        -------
        mse : reconstruction MSE using the final visible probabilities.
        """
        # Positive phase
        _, h0_prob = self.propup(v0)

        # Negative phase
        vk, hk_prob, v_prob = self.gibbs_k(v0, k=int(k))

        # Gradients (classic RBM expectations)
        B = tf.cast(tf.shape(v0)[0], v0.dtype)
        pos = tf.linalg.matmul(tf.transpose(v0), h0_prob) / B
        neg = tf.linalg.matmul(tf.transpose(vk), hk_prob) / B

        dW = pos - neg
        if float(weight_decay) > 0.0:
            dW -= float(weight_decay) * self.W

        dvb = tf.reduce_mean(v0 - vk, axis=0)
        dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)

        # Manual SGD_style updates
        self.W.assign_add(float(lr) * dW)
        self.v_bias.assign_add(float(lr) * dvb)
        self.h_bias.assign_add(float(lr) * dhb)

        # Track reconstruction error against probabilities (more stable than samples)
        return tf.reduce_mean(tf.square(v0 - v_prob))

    @tf.function(jit_compile=False)
    def train_step_mse(self, v0: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
        """
        One optimizer step minimizing reconstruction MSE.

        Returns
        -------
        loss : scalar reconstruction MSE.
        """
        with tf.GradientTape() as tape:
            v_hat = self(v0, training=True)
            loss = tf.reduce_mean(tf.square(v0 - v_hat))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


# =============================================================================
# Convenience builder (seed + eager build)
# =============================================================================
def build_rbm(cfg: RBMConfig) -> RBM:
    """
    Build an RBM from RBMConfig, seeding TF/NumPy (best_effort) and
    creating variables immediately so save/load works consistently.

    This stays artifact_free; it only constructs the model.
    """
    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))
        tf.random.set_seed(int(cfg.seed))

    rbm = RBM(visible_units=int(cfg.visible_units), hidden_units=int(cfg.hidden_units))

    # Eager_build by calling once (ensures variables are initialized and
    # Keras trackables exist consistently before checkpointing).
    _ = rbm(tf.zeros((1, int(cfg.visible_units)), dtype=tf.float32))

    return rbm


# =============================================================================
# Thin wrapper expected by samplers (sample.py)
# =============================================================================
class BernoulliRBM(RBM):
    """
    Wrapper used by samplers and legacy code paths.

    Exposes:
      - sample_h_given_v
      - sample_v_given_h

    Also accepts `visible_dim` / `hidden_dim` naming to match older code.
    """

    def __init__(
        self,
        *,
        visible_dim: Optional[int] = None,
        hidden_dim: int = 256,
        name: str = "rbm",
        **kwargs,
    ) -> None:
        # Back_compat: allow visible_units/hidden_units in kwargs
        if visible_dim is None:
            if "visible_units" in kwargs:
                visible_dim = int(kwargs.pop("visible_units"))
            else:
                raise TypeError("BernoulliRBM requires `visible_dim=` (or `visible_units=`) integer.")
        if "hidden_units" in kwargs:
            hidden_dim = int(kwargs.pop("hidden_units"))

        super().__init__(visible_units=int(visible_dim), hidden_units=int(hidden_dim), name=name)

        # Eager build so load_weights(...) can restore variables immediately.
        _ = self(tf.zeros((1, int(visible_dim)), dtype=tf.float32))

    @tf.function(reduce_retracing=True)
    def sample_h_given_v(self, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample hidden units given visible vector(s).
        Returns (h_sample, h_prob).
        """
        _, h_prob = self.propup(v)
        h_sample = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)
        return h_sample, h_prob

    @tf.function(reduce_retracing=True)
    def sample_v_given_h(self, h: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample visible units given hidden vector(s).
        Returns (v_sample, v_prob).
        """
        _, v_prob = self.propdown(h)
        v_sample = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        return v_sample, v_prob


__all__ = [
    "RBMConfig",
    "RBM",
    "build_rbm",
    "BernoulliRBM",
    "to_float01",
    "binarize01",
    "flatten_images",
    "reshape_to_images",
]