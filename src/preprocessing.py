"""The preprocessing contract — the one thing training and serving must agree on.

Deliberately dependency-light: TensorFlow only, no pandas or scikit-learn. The
serving container imports this module and nothing else from `src`, which keeps
the image lean and means a training-side dependency can never break a deploy.

`src/data.py` imports from here too, so there is exactly one implementation.
"""
from __future__ import annotations

import tensorflow as tf

BINARY_LABELS = {0: "Healthy", 1: "Diseased"}


def preprocess(image: tf.Tensor, img_size: int) -> tf.Tensor:
    """Resize to (img_size, img_size) and return float32 in the 0-255 range.

    NOT rescaled to [0,1]: EfficientNet carries its own normalisation layers and
    expects raw 0-255 input. Dividing by 255 here would silently cost accuracy.
    """
    image = tf.image.resize(image, (img_size, img_size), method="bilinear")
    return tf.cast(image, tf.float32)


def decode_image_bytes(raw: bytes, img_size: int) -> tf.Tensor:
    """Bytes -> model-ready (1, H, W, 3) batch. Used by the API.

    `expand_animations=False` matters: without it an animated GIF decodes to a
    4-D tensor and the resize fails with a shape error that looks unrelated.
    """
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.expand_dims(preprocess(image, img_size), axis=0)