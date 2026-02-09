from __future__ import annotations

import tensorflow as tf


def linear_lambda(iteration: int, max_iterations: int, max_lambda: float = 9.64) -> float:
    """Linear ramp from 0 to ``max_lambda`` over ``max_iterations`` steps."""
    iteration = int(iteration)
    max_iterations = int(max_iterations)
    if iteration >= max_iterations:
        return float(max_lambda)
    return (float(max_lambda) / float(max_iterations)) * float(iteration)


def schedule_lambda(step, max_steps: int, max_lambda: float = 2.0) -> tf.Tensor:
    """Linear ramp used by adversarial models (AB-LSTM style)."""
    step = tf.cast(step, tf.float32)
    max_steps = tf.cast(max_steps, tf.float32)
    max_lambda = tf.cast(max_lambda, tf.float32)
    return tf.minimum((step / max_steps) * max_lambda, max_lambda)
