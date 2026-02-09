import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Conv1DTranspose,
    Reshape,
    LSTM,
    Concatenate,
)

from .layers import GradientReversalLayer


def _copy_like(y):
    """Safe copy for Tensor or numpy array."""
    if tf.is_tensor(y):
        return tf.identity(y)
    return y.copy()


class AEBackbone(tf.keras.Model):
    """Common backbone for autoencoder-based counterfactual models.

    The backbone encodes a covariate sequence ``x`` with an LSTM and an outcome
    sequence ``y`` with a small 1D CNN encoder. A decoder maps
    ``(label, x_embed, z)`` back to an outcome sequence.

    Subclasses implement the latent encoding (e.g., deterministic, VAE, adversarial).
    """

    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        feat_dim: int,
        hidden_layer_sizes,
        dec_dense_units: int = 1600,
    ):
        super().__init__()

        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.latent_dim = int(latent_dim)
        self.feat_dim = int(feat_dim)
        self.seq_len = int(seq_len)
        self.dec_dense_units = int(dec_dense_units)

        if self.dec_dense_units % self.hidden_layer_sizes[-1] != 0:
            raise ValueError(
                f"dec_dense_units={self.dec_dense_units} must be divisible by "
                f"hidden_layer_sizes[-1]={self.hidden_layer_sizes[-1]} "
                "to reshape decoder input cleanly."
            )

        # x branch
        self.lstm_x = LSTM(8)
        self.dense_x_encoder = Dense(8, activation="relu")
        self.dense_x_decoder = Dense(8, activation="relu")

        # join
        self.concat = Concatenate()
        self.dense_concat = Dense(64, activation="relu")

        # y encoder convs
        self.conv1 = Conv1D(
            filters=self.hidden_layer_sizes[0],
            kernel_size=3,
            strides=2,
            activation="relu",
            padding="same",
        )
        self.conv2 = Conv1D(
            filters=self.hidden_layer_sizes[1],
            kernel_size=3,
            strides=2,
            activation="relu",
            padding="same",
        )
        self.flatten = Flatten()

        # decoder
        self.dec_dense = Dense(self.dec_dense_units, name="dec_dense", activation="relu")
        self.dec_reshape = Reshape(
            target_shape=(-1, self.hidden_layer_sizes[-1]),
            name="dec_reshape",
        )
        self.convTr1 = Conv1DTranspose(
            filters=self.hidden_layer_sizes[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )
        self.convTr2 = Conv1DTranspose(
            filters=self.hidden_layer_sizes[-2],
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )
        self.convTr3 = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )
        self.dec_flatten = Flatten(name="dec_flatten")
        self.decoder_dense_final = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")
        self.dec_out_reshape = Reshape(target_shape=(self.seq_len, self.feat_dim))

        # Cached latent representation (set during forward pass).
        self.z = None

    # ------------------ shared pieces ------------------

    def _encode_features(self, label, x_feat, y):
        x_feat = self.dense_x_encoder(x_feat)
        h = self.conv1(y)
        h = self.conv2(h)
        h = self.flatten(h)
        t = self.concat([label, x_feat, h])
        t = self.dense_concat(t)
        return t, x_feat

    def decode(self, label, x_feat, z):
        x_feat = self.dense_x_decoder(x_feat)
        h = self.concat([label, x_feat, z])
        h = self.dec_dense(h)
        h = self.dec_reshape(h)
        h = self.convTr1(h)
        h = self.convTr2(h)
        h = self.convTr3(h)
        h = self.dec_flatten(h)
        h = self.decoder_dense_final(h)
        out = self.dec_out_reshape(h)
        return out

    def reconstruction(self, inputs, out):
        """
        Mean absolute error over the time axis (shape: (batch, feat_dim)).
        """
        return tf.keras.backend.mean(tf.keras.backend.abs(inputs - out), axis=1)

    # ------------------ CF utilities ------------------

    def cf_generation(self, label_real, label_cf, x, y):
        x_feat = self.lstm_x(x)
        z = self.encode(label_real, x_feat, y)
        out = self.decode(label_cf, x_feat, z)
        return out

    def composition(self, labels, x, y, n: int):
        y_ = _copy_like(y)
        for _ in range(int(n)):
            y_ = self.cf_generation(labels, labels, x, y_)
        return y_

    def reversibility(self, label_real, label_cf, x, y, n: int):
        y_ = _copy_like(y)
        for _ in range(int(n)):
            y_cf = self.cf_generation(label_real, label_cf, x, y_)
            y_ = self.cf_generation(label_cf, label_real, x, y_cf)
        return y_

    # subclasses must implement:
    #   encode(...)
    #   call(...)


class CEPAE(AEBackbone):
    """Conditional Entropy Penalized Autoencoder (CEPAE)."""

    def __init__(self, seq_len, latent_dim, feat_dim, hidden_layer_sizes, Lambda=0.1, dec_dense_units=1600):
        super().__init__(
            seq_len=seq_len,
            latent_dim=latent_dim,
            feat_dim=feat_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            dec_dense_units=dec_dense_units,
        )
        # Regularization strength (scheduled externally).
        self.Lambda = tf.Variable(Lambda, trainable=False, dtype=tf.float32, name="lambda")
        self.z_layer = Dense(self.latent_dim, name="z")

    def encode(self, label, x_feat, y):
        t, _ = self._encode_features(label, x_feat, y)
        z = self.z_layer(t)
        return z

    def regularization(self, inputs=None, out=None):
        # Stored for compatibility with existing training code.
        if self.z is None:
            return tf.constant(0.0, dtype=tf.float32)
        z_std = tf.math.reduce_std(self.z, axis=0)
        return self.Lambda * tf.math.reduce_sum(z_std)

    def loss_(self, inputs, out):
        rec = self.reconstruction(inputs, out)
        regu = self.regularization(inputs, out)
        return rec + regu  # broadcast over feature dimension

    def call(self, inputs):
        label, input_x, input_y = inputs
        x_feat = self.lstm_x(input_x)
        z = self.encode(label, x_feat, input_y)
        self.z = z
        out = self.decode(label, x_feat, z)
        return out


class CVAE(AEBackbone):
    """Conditional variational autoencoder (CVAE) baseline."""

    def __init__(self, seq_len, latent_dim, feat_dim, hidden_layer_sizes, recon_weight=200, dec_dense_units=1600):
        super().__init__(
            seq_len=seq_len,
            latent_dim=latent_dim,
            feat_dim=feat_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            dec_dense_units=dec_dense_units,
        )
        self.recon_weight = recon_weight

        # Prior variance for the KL divergence term.
        self.var_prior = 1.0
        det_cov_pz = self.var_prior ** (self.latent_dim)
        self.log_det_cov_pz = tf.math.log(det_cov_pz)

        self.mean_layer = Dense(self.latent_dim, name="z_mean")
        self.logvar_layer = Dense(self.latent_dim, name="z_logvar")

        # Cached distribution parameters (set during forward pass).
        self.mean = None
        self.logvar = None

    def encode(self, label, x_feat, y):
        t, _ = self._encode_features(label, x_feat, y)
        mean = self.mean_layer(t)
        logvar = self.logvar_layer(t)
        return mean, logvar

    def sample(self, mean, logvar):
        eps = tf.random.normal(shape=(tf.shape(mean)[0], self.latent_dim))
        return eps * tf.exp(logvar * 0.5) + mean

    def kl(self, inputs=None, out=None):
        mean = self.mean
        logvar = self.logvar
        var = tf.math.exp(logvar)
        det_cov_qz_x = tf.math.reduce_prod(var, axis=1)

        kl = 0.5 * (
            self.log_det_cov_pz
            - tf.math.log(det_cov_qz_x)
            - self.latent_dim
            + tf.math.reduce_sum((mean * mean / self.var_prior), axis=1)
            + tf.math.reduce_sum(var / self.var_prior, axis=1)
        )
        return kl

    def loss_(self, inputs, out):
        rec = tf.reshape(self.reconstruction(inputs, out), [-1, 1])
        kl = tf.reshape(self.kl(inputs, out), [-1, 1])
        return (self.recon_weight) * rec + kl

    def cf_generation(self, label_real, label_cf, x, y):
        x_feat = self.lstm_x(x)
        mean, logvar = self.encode(label_real, x_feat, y)
        z = self.sample(mean, logvar)
        out = self.decode(label_cf, x_feat, z)
        return out

    def call(self, inputs):
        label, input_x, input_y = inputs
        x_feat = self.lstm_x(input_x)
        self.mean, self.logvar = self.encode(label, x_feat, input_y)
        z = self.sample(self.mean, self.logvar)
        self.z = z
        out = self.decode(label, x_feat, z)
        return out


class CAAE(AEBackbone):
    """Conditional adversarial autoencoder (CAAE) baseline."""

    def __init__(self, seq_len, latent_dim, feat_dim, hidden_layer_sizes, dec_dense_units=1600):
        super().__init__(
            seq_len=seq_len,
            latent_dim=latent_dim,
            feat_dim=feat_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            dec_dense_units=dec_dense_units,
        )

        self.z_layer = Dense(self.latent_dim, name="z")

        # domain head
        self.grl = GradientReversalLayer()
        self.domain_dense1 = Dense(32, activation="relu")
        self.domain_dense2 = Dense(1, activation="sigmoid")
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def encode(self, label, x_feat, y):
        t, _ = self._encode_features(label, x_feat, y)
        return self.z_layer(t)

    def domain_classifier(self, z, lambd=1.0):
        z_rev = self.grl(z, lambd=lambd)
        h = self.domain_dense1(z_rev)
        return self.domain_dense2(h)

    def domain_loss(self, label, z, lambd=1.0):
        label = tf.reshape(label, (-1, 1))
        pred = self.domain_classifier(z, lambd=lambd)
        loss = self.bce(label, pred)
        return loss, pred

    def call(self, inputs):
        label, input_x, input_y = inputs
        x_feat = self.lstm_x(input_x)
        z = self.encode(label, x_feat, input_y)
        self.z = z
        out = self.decode(label, x_feat, z)
        return out
