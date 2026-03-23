# src/gencysynth/models/gan/variants/dcgan/model.py
"""
GenCyberSynth — GAN Family — DCGAN Variant (Conditional)

Location (variant contract)
---------------------------
This module lives under the standardized variant folder:

    src/gencysynth/models/gan/variants/dcgan/
        ├── model.py        <-- (this file) model builders only (NO file I/O)
        ├── train.py        training loop + checkpoints/logging (writes artifacts)
        ├── sample.py       sampling/synthesis (writes images/npz/etc.)
        └── defaults.yaml   optional default hyperparameters for this variant

What this module is responsible for
-----------------------------------
Pure model construction and compilation:
    - build_generator(...)
    - build_discriminator(...)
    - build_models(...) -> {"generator": G, "discriminator": D, "gan": GAN}

What this module MUST NOT do
-------------------------------
- No dataset loading
- No writing files (checkpoints, images, logs)
- No knowledge of run directories / artifact paths
Those responsibilities belong to:
    - variant train.py / sample.py
    - orchestration (run spec + manifest)
    - utils/paths.py, utils/run_id.py, orchestration/manifest.py

Variant summary
---------------
This is a *conditional* DCGAN:
- Generator takes:
    z : (latent_dim,)
    y : (num_classes,) one_hot label
  and outputs:
    x_fake : img_shape, with tanh activation -> values in [-1, 1]

- Discriminator takes:
    x : img_shape
    y : (num_classes,) one_hot label
  and outputs:
    p_real : scalar sigmoid probability

Conditioning strategy
---------------------
We condition both G and D by projecting the one_hot label into a spatial map,
then concatenating it channel_wise with feature maps / images. This keeps the
conditioning mechanism consistent for grayscale or RGB images.

Compatibility notes
-------------------
- Uses tf.keras (Keras 3 compatible).
- LeakyReLU uses `negative_slope=0.2` to avoid deprecated `alpha`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

# Keras 3: use standard (non_legacy) Adam
Adam = tf.keras.optimizers.Adam


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
def build_generator(
    latent_dim: int = 100,
    num_classes: int = 9,
    img_shape: Tuple[int, int, int] = (40, 40, 1),
) -> tf.keras.Model:
    """
    Build the conditional generator (G).

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent noise vector z.
    num_classes:
        Number of classes; y is expected to be one_hot of shape (num_classes,).
    img_shape:
        Output image shape (H, W, C). For USTC_TFC2016 this is typically (40, 40, 1).

    Returns
    -------
    tf.keras.Model
        A Keras model mapping (z, y_onehot) -> x_fake in [-1, 1] via tanh.

    Notes
    -----
    - z is projected to a small spatial feature map (5x5x256)
    - y is projected to a 5x5x1 conditioning map
    - maps are concatenated and upsampled to HxW
    """
    H, W, C = img_shape  # kept for readability / future sanity checks

    # Inputs
    z_in = layers.Input(shape=(latent_dim,), name="z")
    y_in = layers.Input(shape=(num_classes,), name="y_onehot")

    # ---- Project latent vector into a low_resolution feature grid ----
    x = layers.Dense(5 * 5 * 256, use_bias=False, name="gen_dense_z")(z_in)
    x = layers.BatchNormalization(name="gen_bn_z")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="gen_lrelu_z")(x)
    x = layers.Reshape((5, 5, 256), name="gen_reshape_z")(x)

    # ---- Project label to a small spatial map and concatenate ----
    # We use a single_channel label map so conditioning works for grayscale or RGB.
    y_map = layers.Dense(5 * 5 * 1, use_bias=False, name="gen_dense_y")(y_in)
    y_map = layers.Reshape((5, 5, 1), name="gen_reshape_y")(y_map)

    # Concatenate noise feature map with label conditioning map: (5, 5, 256+1)
    x = layers.Concatenate(axis=-1, name="gen_concat_zy")([x, y_map])

    # ---- Upsampling blocks to reach the target resolution ----
    # 5x5 -> 10x10
    x = layers.UpSampling2D(name="gen_up1")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="gen_conv1")(x)
    x = layers.BatchNormalization(name="gen_bn1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="gen_lrelu1")(x)

    # 10x10 -> 20x20
    x = layers.UpSampling2D(name="gen_up2")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="gen_conv2")(x)
    x = layers.BatchNormalization(name="gen_bn2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="gen_lrelu2")(x)

    # 20x20 -> 40x40
    x = layers.UpSampling2D(name="gen_up3")(x)

    # Final convolution to match channel count C; tanh enforces [-1, 1]
    out = layers.Conv2D(
        filters=C,
        kernel_size=3,
        padding="same",
        activation="tanh",
        use_bias=False,
        name="gen_out",
    )(x)

    return models.Model(inputs=[z_in, y_in], outputs=out, name="gan.dcgan.generator")


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------
def build_discriminator(
    img_shape: Tuple[int, int, int] = (40, 40, 1),
    num_classes: int = 9,
) -> tf.keras.Model:
    """
    Build the conditional discriminator (D).

    Parameters
    ----------
    img_shape:
        Input image shape (H, W, C).
    num_classes:
        Number of classes; y is expected to be one_hot of shape (num_classes,).

    Returns
    -------
    tf.keras.Model
        A Keras model mapping (x, y_onehot) -> p_real in [0, 1] (sigmoid).

    Conditioning strategy
    ---------------------
    - Project y_onehot to an HxW single_channel map
    - Concatenate with x along channels to form (H, W, C+1)
    """
    H, W, C = img_shape

    # Inputs
    x_in = layers.Input(shape=img_shape, name="x")
    y_in = layers.Input(shape=(num_classes,), name="y_onehot")

    # Project y into an HxW conditioning map (single channel)
    y_map = layers.Dense(H * W, use_bias=False, name="disc_dense_y")(y_in)
    y_map = layers.Reshape((H, W, 1), name="disc_reshape_y")(y_map)

    # Concatenate image with conditioning map: (H, W, C+1)
    xy = layers.Concatenate(axis=-1, name="disc_concat_xy")([x_in, y_map])

    # Standard DCGAN_style conv downsampling
    x = layers.Conv2D(64, 4, strides=2, padding="same", name="disc_conv1")(xy)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu1")(x)
    x = layers.Dropout(0.3, name="disc_drop1")(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", name="disc_conv2")(x)
    x = layers.BatchNormalization(name="disc_bn2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu2")(x)
    x = layers.Dropout(0.3, name="disc_drop2")(x)

    x = layers.Flatten(name="disc_flatten")(x)
    out = layers.Dense(1, activation="sigmoid", name="disc_out")(x)

    return models.Model(inputs=[x_in, y_in], outputs=out, name="gan.dcgan.discriminator")


# ---------------------------------------------------------------------
# Convenience builder: Generator + Discriminator + Combined GAN
# ---------------------------------------------------------------------
def build_models(
    latent_dim: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    lr: float = 2e_4,
    beta_1: float = 0.5,
) -> Dict[str, tf.keras.Model]:
    """
    Build and compile the model trio for GAN training:

    Returns a dict with:
        - "generator": G  (not compiled; trained via the combined model)
        - "discriminator": D (compiled)
        - "gan": combined GAN = D(G(z,y), y) (compiled with D frozen)

    Important design rule
    ---------------------
    This function compiles models but still performs NO file I/O and assumes
    nothing about run directories. Artifact paths and logging live elsewhere.

    Parameters
    ----------
    latent_dim, num_classes, img_shape:
        Shape controls for the conditional DCGAN.
    lr, beta_1:
        Adam optimizer hyperparameters (DCGAN standard defaults).

    Notes
    -----
    - D is trained with binary cross_entropy + accuracy metric.
    - GAN is trained with binary cross_entropy with D frozen.
    """
    # Build component models
    G = build_generator(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape)
    D = build_discriminator(img_shape=img_shape, num_classes=num_classes)

    # Optimizers (Keras 3 compatible)
    d_opt = Adam(learning_rate=lr, beta_1=beta_1)
    g_opt = Adam(learning_rate=lr, beta_1=beta_1)

    # Compile discriminator for standalone training updates
    D.compile(optimizer=d_opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Build combined GAN: D(G(z,y), y) with D frozen
    z_in = layers.Input(shape=(latent_dim,), name="z_in")
    y_in = layers.Input(shape=(num_classes,), name="y_in")

    fake = G([z_in, y_in])

    # Freeze D weights for generator updates through the combined model
    D.trainable = False
    validity = D([fake, y_in])

    GAN = models.Model(inputs=[z_in, y_in], outputs=validity, name="gan.dcgan.combined")
    GAN.compile(optimizer=g_opt, loss="binary_crossentropy")

    return {"generator": G, "discriminator": D, "gan": GAN}


__all__ = ["build_generator", "build_discriminator", "build_models"]
