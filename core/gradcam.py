from __future__ import annotations

import numpy as np


def get_gradcam(img_tensor, base_model, head_layers) -> np.ndarray:
    import tensorflow as tf

    if base_model is None:
        return np.zeros((12, 12))
    img_t = tf.cast(img_tensor, tf.float32)
    conv_out = base_model(img_t, training=False)
    conv_var = tf.Variable(conv_out)
    with tf.GradientTape() as tape:
        tape.watch(conv_var)
        x = conv_var
        for layer in head_layers:
            x = layer(x)
        score = x[:, tf.argmax(x[0])]
    grads = tape.gradient(score, conv_var)
    if grads is None:
        return np.zeros((12, 12))
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-08)
    return heatmap.numpy()
