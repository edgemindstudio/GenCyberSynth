# src/gencysynth/models/diffusion/variants/c_ddpm/model.py
"""
GenCyberSynth — Diffusion family — c_DDPM variant (Conditional) — Model Builders
===============================================================================

RULE A (Scalable artifact policy)
---------------------------------
This file is a **pure model_construction module**.

It MUST NOT:
  - read or write artifacts
  - know about run directories / checkpoints / manifests
  - load datasets

All I/O (checkpoints, samples, logs) belongs to:
  - variants/c_ddpm/train.py   (writes run_scoped checkpoints + tensorboard)
  - variants/c_ddpm/samply.py  (reads run_scoped checkpoints, writes run_scoped samples + manifest)
  - orchestration/context.py   (resolves dataset_id / model_tag / run_id paths)

What this module provides
-------------------------
- SinusoidalTimeEmbedding: Keras layer mapping timesteps -> embeddings
- build_diffusion_model(...): compact conditional UNet_like εθ predictor
  The returned model is **compiled** (MSE on noise, Adam) and ready for training
  or weight loading.

Conventions
-----------
- Images are channels_last (H, W, C).
- Training inputs are noisy images x_t in the same scale as real data.
  In GenCyberSynth, the diffusion training loop expects images in **[0, 1]**.
- Labels are one_hot vectors of length `num_classes`.
- Weight filenames should end with `.weights.h5` (Keras 3 friendly).

Notes on conditioning
---------------------
We condition via:
  - sinusoidal time embedding t -> vector
  - label embedding y -> vector
  - concatenate + project -> cond vector
  - broadcast cond over spatial dims -> (B, H, W, D)
  - concatenate with noisy image channels -> UNet backbone input

This matches the simple, portable conditioning strategy used across variants.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


# =============================================================================
# Time embedding
# =============================================================================
class SinusoidalTimeEmbedding(layers.Layer):
    """
    Sinusoidal timestep embedding as in Transformer/DDPM literature.

    Input
    -----
    t : Tensor
        int32/float32 tensor of shape (B,) or (B,1) or scalar per_batch.

    Output
    ------
    emb : Tensor
        float32 tensor of shape (B, dim)

    Implementation details
    ----------------------
    - Uses sin/cos pairs for an even_sized embedding (2 * half_dim).
    - If dim is odd, uses a Dense projection to reach exact dim.
    - Handles tiny dims safely (avoids divide_by_zero when half_dim == 1).
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.half_dim = max(1, self.dim // 2)

        # If dim is odd, the sin/cos concat yields 2*half_dim < dim.
        # We project to exact dim for consistent downstream shapes.
        self.proj = None
        if self.dim != 2 * self.half_dim:
            self.proj = layers.Dense(self.dim, name="time_proj")

    def call(self, t):
        # Flatten to (B,)
        t = tf.reshape(t, (-1,))
        t = tf.cast(t, tf.float32)

        # Frequencies: exp(-log(10000) * i / (half_dim - 1)), i=0..half_dim_1
        denom = tf.cast(tf.maximum(self.half_dim - 1, 1), tf.float32)
        freqs = tf.exp(
            tf.range(self.half_dim, dtype=tf.float32)
            * -(tf.math.log(10000.0) / denom)
        )  # (half_dim,)

        # Outer product -> (B, half_dim)
        args = tf.expand_dims(t, -1) * tf.expand_dims(freqs, 0)
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (B, 2*half_dim)

        if self.proj is not None:
            emb = self.proj(emb)

        return emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg


# =============================================================================
# Small building blocks
# =============================================================================
def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """
    A compact "UNet_style" block:
      Conv2D -> LayerNorm -> Swish -> Conv2D -> LayerNorm -> Swish

    LayerNorm (instead of BatchNorm) is typically more stable at small batch sizes
    and on mixed hardware (CPU/GPU/Apple Silicon), which fits GenCyberSynth usage.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.LayerNormalization(name=f"{name}_ln1")(x)
    x = layers.Activation("swish", name=f"{name}_act1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.LayerNormalization(name=f"{name}_ln2")(x)
    x = layers.Activation("swish", name=f"{name}_act2")(x)
    return x


def _broadcast_to_spatial(emb: tf.Tensor, h: int, w: int, name_prefix: str = "cond") -> tf.Tensor:
    """
    Convert a vector embedding (B, D) into a spatial map (B, H, W, D).

    We wrap the ops in Lambda layers to keep everything KerasTensor_safe.

    Parameters
    ----------
    emb : Tensor
        Shape (B, D)
    h, w : int
        Target spatial dimensions.
    name_prefix : str
        Layer naming prefix for clean model graphs.
    """
    x = layers.Lambda(
        lambda e: tf.expand_dims(tf.expand_dims(e, 1), 1),
        name=f"{name_prefix}_expand",
    )(emb)  # (B,1,1,D)

    x = layers.Lambda(
        lambda e: tf.tile(e, [1, h, w, 1]),
        name=f"{name_prefix}_tile",
    )(x)  # (B,H,W,D)

    return x


# =============================================================================
# Public builder
# =============================================================================
def build_diffusion_model(
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    base_filters: int = 64,
    depth: int = 2,
    time_emb_dim: int = 128,
    learning_rate: float = 2e_4,
    beta_1: float = 0.9,
) -> tf.keras.Model:
    """
    Build a compact conditional UNet that predicts noise ε from (x_t, t, y).

    Inputs
    ------
    noisy_image : (H, W, C) float32
        The noisy image x_t (same scale as training data, typically [0,1]).
    class_label : (num_classes,) float32
        One_hot class label.
    timestep : () int32
        Diffusion timestep t.

    Output
    ------
    predicted_noise : (H, W, C) float32
        ε̂θ(x_t, t, y), no activation.

    Notes
    -----
    - Enforces that H and W are divisible by 2**depth (required by down/upsampling).
    - Model is compiled with Adam + MSE (standard ε_prediction objective).
    - This function performs **no I/O** and knows nothing about artifacts paths.
    """
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")

    H, W, C = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
    stride_total = 2**depth
    if (H % stride_total) != 0 or (W % stride_total) != 0:
        raise ValueError(
            f"img_shape spatial dims must be divisible by 2**depth ({stride_total}). "
            f"Got H={H}, W={W}, depth={depth}."
        )

    # -----------------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------------
    noisy_in = layers.Input(shape=(H, W, C), name="noisy_image")
    y_in = layers.Input(shape=(int(num_classes),), name="class_label")
    t_in = layers.Input(shape=(), dtype=tf.int32, name="timestep")

    # -----------------------------------------------------------------
    # Conditioning: time embedding + label embedding -> spatial map
    # -----------------------------------------------------------------
    # Time embedding (sinusoidal -> MLP)
    t_emb = SinusoidalTimeEmbedding(int(time_emb_dim), name="time_embed")(t_in)
    t_emb = layers.Dense(int(time_emb_dim), activation="swish", name="time_mlp1")(t_emb)
    t_emb = layers.Dense(int(time_emb_dim), activation="swish", name="time_mlp2")(t_emb)

    # Label embedding (one_hot -> MLP)
    y_emb = layers.Dense(int(time_emb_dim), activation="swish", name="label_mlp1")(y_in)
    y_emb = layers.Dense(int(time_emb_dim), activation="swish", name="label_mlp2")(y_emb)

    # Combine and project
    cond = layers.Concatenate(name="cond_concat")([t_emb, y_emb])  # (B, 2*D)
    cond = layers.Dense(int(time_emb_dim), activation="swish", name="cond_proj")(cond)  # (B, D)

    # Broadcast to spatial map and concatenate with noisy image
    cond_spatial = _broadcast_to_spatial(cond, H, W, name_prefix="cond")  # (B,H,W,D)
    x = layers.Concatenate(name="input_concat")([noisy_in, cond_spatial])  # (B,H,W,C+D)

    # -----------------------------------------------------------------
    # UNet backbone (down -> bottleneck -> up) with skip connections
    # -----------------------------------------------------------------
    skips: list[tf.Tensor] = []
    filters = int(base_filters)

    # Down path: convs + strided conv downsample each level
    for d in range(depth):
        x = _conv_block(x, filters, name=f"down{d}")
        skips.append(x)  # save skip at this resolution
        x = layers.Conv2D(filters, 3, strides=2, padding="same", name=f"down{d}_ds")(x)
        filters *= 2

    # Bottleneck at smallest resolution
    x = _conv_block(x, filters, name="bottleneck")

    # Up path: upsample + concat skip + conv block
    for d, skip in enumerate(reversed(skips)):
        filters //= 2
        x = layers.UpSampling2D(size=2, interpolation="nearest", name=f"up{d}_us")(x)
        x = layers.Concatenate(name=f"up{d}_skip")([x, skip])
        x = _conv_block(x, filters, name=f"up{d}")

    # Output head: predict noise (no activation)
    out = layers.Conv2D(C, 1, padding="same", name="pred_noise")(x)

    model = models.Model(inputs=[noisy_in, y_in, t_in], outputs=out, name="diffusion.c_ddpm.unet")

    # -----------------------------------------------------------------
    # Compile (standard ε_prediction objective)
    # -----------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate), beta_1=float(beta_1)),
        loss="mse",
    )
    return model


__all__ = ["SinusoidalTimeEmbedding", "build_diffusion_model"]
