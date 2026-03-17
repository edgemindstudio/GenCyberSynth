# src/gencysynth/models/autoregressive/variants/c_pixelcnnpp/model.py

"""
GenCyberSynth — Autoregressive — c_pixelcnnpp — Model
=====================================================

This module defines the *model_only* portion of the conditional PixelCNN_style
baseline. Under Rule A, this file should NOT do any artifact I/O. All reading
and writing of checkpoints / synthetic dumps belongs in:
  - train.py
  - sample.py
  - pipeline.py

What this module provides
-------------------------
- MaskedConv2D: PixelCNN_style masked convolution layer (Mask A/B).
- build_conditional_pixelcnn(...): Keras model returning per_pixel probabilities.

Conventions
-----------
- Images are channels_last: (H, W, C)
- Inputs x are expected in [0,1] for training with pixelwise BCE.
- Conditioning is class_based: y is one_hot (num_classes,)

Notes
-----
- The output head uses sigmoid -> probabilities in [0,1] (not logits).
- For grayscale data we default to output channels = C (usually 1), but this
  is configurable (out_channels).
- Label conditioning is implemented as a broadcast label map (H,W,label_channels)
  concatenated with the image input. This is a simple, stable baseline.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers


# =============================================================================
# Mask helpers
# =============================================================================
def _make_causal_mask(
    kh: int,
    kw: int,
    in_ch: int,
    out_ch: int,
    mask_type: str,
) -> np.ndarray:
    """
    Create a PixelCNN causal mask for a Conv2D kernel.

    Mask meaning
    ------------
    - Mask "A": the center pixel is masked OUT (no access to current pixel).
      Used in the first layer to avoid peeking at x(i,j).
    - Mask "B": the center pixel is allowed (access to current pixel), but still
      forbids access to "future" pixels.

    Kernel layout: (kh, kw, in_ch, out_ch)
    """
    mtype = str(mask_type).upper()
    if mtype not in ("A", "B"):
        raise ValueError("mask_type must be 'A' or 'B'")

    m = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
    ch, cw = kh // 2, kw // 2

    # Zero all rows strictly below the center row (future rows).
    m[ch + 1 :, :, :, :] = 0.0

    # Zero all columns strictly to the right of the center in the center row (future cols).
    m[ch, cw + 1 :, :, :] = 0.0

    # For mask "A", also zero the center pixel itself (no access to current pixel).
    if mtype == "A":
        m[ch, cw, :, :] = 0.0

    return m


class MaskedConv2D(layers.Layer):
    """
    PixelCNN_style masked conv using explicit kernel/bias weights.

    Why explicit weights?
    ---------------------
    This avoids some edge cases where layer wrapping can change variable names
    or shapes unexpectedly. Using explicit `kernel` and `bias` keeps checkpoint
    compatibility stable across minor refactors (helpful on long HPC projects).

    Input shape:  (B, H, W, Cin)
    Output shape: (B, H, W, filters)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int | Tuple[int, int],
        mask_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.mask_type = str(mask_type).upper()
        if self.mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")

        # Internal cached constant mask; built in build().
        self._mask = None

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])

        # Precompute constant mask.
        mask = _make_causal_mask(kh, kw, in_ch, self.filters, self.mask_type)

        # Trainable kernel/bias.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(kh, kw, in_ch, self.filters),
            initializer=initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )

        self._mask = tf.constant(mask, dtype=self.dtype or tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        # Apply the causal mask on_the_fly.
        k = self.kernel * self._mask
        y = tf.nn.conv2d(inputs, k, strides=1, padding="SAME")
        y = tf.nn.bias_add(y, self.bias)
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "mask_type": self.mask_type,
            }
        )
        return cfg


# =============================================================================
# Model builder
# =============================================================================
def build_conditional_pixelcnn(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    *,
    filters: int = 64,
    masked_layers: int = 6,
    label_channels: int = 1,
    out_channels: Optional[int] = None,
) -> tf.keras.Model:
    """
    Build a compact conditional PixelCNN baseline.

    Architecture
    ------------
    - Input: image x (H,W,C), one_hot label y (K,)
    - Conditioning: y -> Dense -> (H,W,label_channels) label map, concat with x
    - Masked stack:
        * Mask A conv (7x7) then ReLU
        * `masked_layers` times: Mask B conv (3x3) then ReLU
    - 1x1 projection + sigmoid output head

    Parameters
    ----------
    img_shape : (H,W,C)
    num_classes : number of classes (one_hot length)
    filters : hidden channels in the PixelCNN stack
    masked_layers : number of Mask_B layers
    label_channels : channels in the broadcast label map
    out_channels : output channels for probs; defaults to input C

    Returns
    -------
    tf.keras.Model
        Inputs:  [image, onehot]
        Output:  probs in [0,1], shape (H,W,out_channels)

    Notes
    -----
    - This model returns probabilities, not logits. Training should use BCE with
      from_logits=False.
    - For multi_channel inputs, set out_channels=C (default) to predict each
      channel; for strict binary grayscale, C=1 is typical.
    """
    H, W, C = img_shape
    K = int(num_classes)
    if K <= 1:
        raise ValueError(f"num_classes must be >= 2 for conditional modeling; got {K}")

    out_ch = int(out_channels) if out_channels is not None else int(C)

    # -------------------- Inputs --------------------
    x_in = layers.Input(shape=(H, W, C), name="image")          # image in [0,1]
    y_in = layers.Input(shape=(K,), name="onehot")              # one_hot label

    # -------------------- Label conditioning --------------------
    # Project label to a spatial map (simple but effective baseline).
    # Shape: (B, H*W*label_channels) -> reshape -> (B,H,W,label_channels)
    lab = layers.Dense(H * W * label_channels, activation="relu", name="label_proj")(y_in)
    lab = layers.Reshape((H, W, label_channels), name="label_map")(lab)

    # Concatenate conditioning map with image.
    h = layers.Concatenate(axis=-1, name="concat_img_label")([x_in, lab])

    # -------------------- Masked conv stack --------------------
    h = MaskedConv2D(filters, kernel_size=7, mask_type="A", name="maskedA_7x7")(h)
    h = layers.ReLU(name="relu_A")(h)

    for i in range(1, int(masked_layers) + 1):
        h = MaskedConv2D(filters, kernel_size=3, mask_type="B", name=f"maskedB_{i}")(h)
        h = layers.ReLU(name=f"relu_B_{i}")(h)

    # -------------------- Output head --------------------
    h = layers.Conv2D(filters, kernel_size=1, padding="same", activation="relu", name="proj_1x1")(h)
    out = layers.Conv2D(out_ch, kernel_size=1, padding="same", activation="sigmoid", name="probs")(h)

    return models.Model(inputs=[x_in, y_in], outputs=out, name="ConditionalPixelCNN")


__all__ = ["MaskedConv2D", "build_conditional_pixelcnn"]