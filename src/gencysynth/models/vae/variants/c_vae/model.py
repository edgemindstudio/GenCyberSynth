# src/gencysynth/models/vae/variants/c_vae/model.py
"""
Rule A — Conditional VAE (cVAE) model builders.

Important: This file is intentionally *artifact_agnostic*.
- Under Rule A, *pipelines / train / sample* decide where artifacts live.
- This module only builds Keras models; it never reads/writes checkpoints.

What this provides
------------------
- Sampling layer (reparameterization trick).
- build_encoder(): (image, onehot) -> (z_mean, z_log_var, z)
- build_decoder(): (z, onehot) -> reconstructed image in [-1, 1] (tanh)
- ConditionalVAE: subclassed Keras Model with train_step/test_step (Keras 3 friendly)
- build_models(): convenience builder returning {"encoder","decoder","vae"}

Conventions
-----------
- Training images in [-1, 1] (tanh decoder convention).
- Labels are one_hot of length `num_classes`.
- Keras 3: checkpoint filenames must end with `.weights.h5` when using save_weights().

Why no artifacts here?
----------------------
Keeping model_building pure makes the repository scalable across datasets and
runs: path decisions belong to config normalization + pipelines, not layers.
"""

from __future__ import annotations

from typing import Dict, Tuple
import math

import tensorflow as tf
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras import ops as K  # Keras 3 math ops (safe for KerasTensors)


# -----------------------------------------------------------------------------
# Reparameterization (Sampling) + math helpers
# -----------------------------------------------------------------------------
class Sampling(layers.Layer):
    """
    Reparameterization trick:
        z = mu + exp(0.5 * log_var) * eps
    where eps ~ N(0, I).
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # Numerical stability guard: extremely large/small log_var can explode gradients.
        z_log_var = K.clip(z_log_var, -10.0, 10.0)

        eps = keras.random.normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * eps


def _broadcast_labels(img: K.Tensor, onehot: K.Tensor) -> K.Tensor:
    """
    Broadcast labels (B, C) -> spatial label_map (B, H, W, C),
    matching the image spatial dimensions. This lets the encoder "see" the class.

    Uses Keras ops so it works in graph mode with KerasTensors.
    """
    y = K.expand_dims(onehot, axis=1)   # (B, 1, C)
    y = K.expand_dims(y, axis=1)        # (B, 1, 1, C)
    ones = K.ones_like(img[..., :1])    # (B, H, W, 1)
    return y * ones                     # (B, H, W, C)


def _kl_per_example(z_mean: K.Tensor, z_log_var: K.Tensor) -> K.Tensor:
    """
    KL divergence per sample:
        KL( N(mu, sigma) || N(0, I) )
    Returns shape (B,).
    """
    return -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_encoder(
    *,
    img_shape: Tuple[int, int, int],
    latent_dim: int,
    num_classes: int,
) -> Model:
    """
    Conditional encoder:
      inputs:  x in [-1,1], shape (H,W,C)
               y one_hot,  shape (K,)
      output:  z_mean, z_log_var, z
    """
    x_in = layers.Input(shape=img_shape, name="image_input")
    y_in = layers.Input(shape=(num_classes,), name="label_input")

    # Broadcast label over spatial dimensions and concatenate with image.
    y_map = layers.Lambda(
        lambda t: _broadcast_labels(t[0], t[1]),
        name="label_broadcast",
    )([x_in, y_in])

    x = layers.Concatenate(name="concat_img_label")([x_in, y_map])

    # A small, stable conv stack
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])

    return Model([x_in, y_in], [z_mean, z_log_var, z], name="cEncoder")


def build_decoder(
    *,
    latent_dim: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
) -> Model:
    """
    Conditional decoder:
      inputs:  z shape (latent_dim,)
               y one_hot shape (K,)
      output:  reconstructed image in [-1, 1] with tanh activation
    """
    H, W, C = img_shape

    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    y_in = layers.Input(shape=(num_classes,), name="label_input")

    x = layers.Concatenate(name="concat_z_label")([z_in, y_in])

    # Choose a small feature map size, then upsample to exact HxW.
    fh, fw = max(1, math.ceil(H / 4)), max(1, math.ceil(W / 4))
    ch = 64

    x = layers.Dense(fh * fw * ch, activation="relu")(x)
    x = layers.Reshape((fh, fw, ch))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    # Guarantee exact output size (helps when H/W aren't divisible by 4).
    x = layers.Resizing(H, W, interpolation="bilinear")(x)

    x_out = layers.Conv2D(C, 3, padding="same", activation="tanh", name="x_recon")(x)

    return Model([z_in, y_in], x_out, name="cDecoder")


# -----------------------------------------------------------------------------
# Subclassed VAE with custom train/test step (Keras 3 friendly)
# -----------------------------------------------------------------------------
class ConditionalVAE(Model):
    """
    Subclassed conditional VAE wrapper.

    - compute_losses() computes recon + KL and returns scalars.
    - train_step/test_step update metrics and (train_step) applies grads.
    """

    def __init__(self, encoder: Model, decoder: Model, beta_kl: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_kl = float(beta_kl)

        # Metrics (Keras will reset them every epoch automatically)
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        """
        Forward pass:
          x_recon = decoder(z, y)
        """
        x, y = inputs
        z_mean, z_log_var, z = self.encoder([x, y], training=training)
        x_recon = self.decoder([z, y], training=training)
        return x_recon

    def compute_losses(self, x, y):
        """
        Compute scalar losses for a batch.
        """
        z_mean, z_log_var, z = self.encoder([x, y], training=True)
        x_recon = self.decoder([z, y], training=True)

        # Reconstruction MSE over pixels
        recon_per_ex = K.mean(K.square(x - x_recon), axis=(1, 2, 3))  # (B,)
        recon_loss = K.mean(recon_per_ex)

        # KL divergence
        kl_per_ex = _kl_per_example(z_mean, z_log_var)
        kl_loss = K.mean(kl_per_ex)

        total = recon_loss + self.beta_kl * kl_loss
        return total, recon_loss, kl_loss

    def train_step(self, data):
        """
        Keras training step. Expects dataset items:
            (x_batch, y_onehot_batch)
        """
        x, y = data
        with tf.GradientTape() as tape:
            total, recon, kl = self.compute_losses(x, y)

        grads = tape.gradient(total, self.trainable_variables)
        grads_vars = [(g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(grads_vars)

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        total, recon, kl = self.compute_losses(x, y)

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# -----------------------------------------------------------------------------
# Convenience builder
# -----------------------------------------------------------------------------
def build_models(
    *,
    img_shape: Tuple[int, int, int],
    latent_dim: int,
    num_classes: int,
    lr: float = 2e_4,
    beta_1: float = 0.5,
    beta_kl: float = 1.0,
) -> Dict[str, Model]:
    """
    Build encoder/decoder and compile a `ConditionalVAE` wrapper.

    Returns a dict:
      {"encoder": enc, "decoder": dec, "vae": vae}

    Notes
    -----
    - We compile `vae` (optimizer) for convenience; training in our repo uses
      the custom loop in pipeline.py for Rule A artifact control.
    - The decoder is *not* compiled separately here to avoid duplicated/unused
      optimizers in memory; you can still call decoder.predict(...) anytime.
    """
    enc = build_encoder(img_shape=img_shape, latent_dim=latent_dim, num_classes=num_classes)
    dec = build_decoder(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape)

    vae = ConditionalVAE(enc, dec, beta_kl=beta_kl, name="cVAE")
    vae.compile(optimizer=Adam(learning_rate=lr, beta_1=beta_1))
    return {"encoder": enc, "decoder": dec, "vae": vae}


__all__ = [
    "Sampling",
    "build_encoder",
    "build_decoder",
    "ConditionalVAE",
    "build_models",
]