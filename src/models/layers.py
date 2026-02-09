import tensorflow as tf


@tf.custom_gradient
def _gradient_reverse(x, lambd):
    """Identity forward pass; multiplies gradient by -lambda."""
    lambd = tf.cast(lambd, x.dtype)

    def grad(dy):
        return -lambd * dy, tf.zeros_like(lambd)

    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer (GRL).

    Call: grl(x, lambd)
      - forward: returns x
      - backward: multiplies upstream gradient by -lambd
    """

    def call(self, x, lambd=1.0):
        lambd = tf.convert_to_tensor(lambd, dtype=x.dtype)
        return _gradient_reverse(x, lambd)
