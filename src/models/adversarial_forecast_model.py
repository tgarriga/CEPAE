import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Concatenate

from .layers import GradientReversalLayer
from ..utils.schedules import schedule_lambda


class AdversarialForecastModel(tf.keras.Model):
    def __init__(self, pred_steps: int = 1):
        super().__init__()
        # Forecasting backbone
        self.lstm1 = LSTM(32, return_sequences=True)
        self.lstm2 = LSTM(32)
        self.concat = Concatenate()
        self.hidden = Dense(32, activation="relu")
        self.forecast_out = Dense(pred_steps)

        # Domain discriminator
        self.grl = GradientReversalLayer()
        self.dom_dense1 = Dense(32, activation="relu")
        self.dom_dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs, *, lambd=1.0, training=False, return_latent=False):
        event, ts = inputs
        h = self.lstm2(self.lstm1(ts, training=training), training=training)
        y_hat = self.forecast_out(self.hidden(self.concat([event, h])), training=training)
        return (y_hat, h) if return_latent else y_hat

    def classify_domain(self, h, lambd):
        reversed_h = self.grl(h, lambd=lambd)
        return self.dom_dense2(self.dom_dense1(reversed_h))


class Trainer:
    def __init__(
        self,
        model: AdversarialForecastModel,
        optimizer: tf.keras.optimizers.Optimizer,
        max_steps: int,
        max_lambda: float = 2.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_steps = int(max_steps)
        self.max_lambda = float(max_lambda)

        self.lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.mae = tf.keras.losses.MeanAbsoluteError()

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_pred_loss = tf.keras.metrics.Mean(name="train_pred_loss")
        self.train_domain_loss = tf.keras.metrics.Mean(name="train_domain_loss")
        self.train_domain_acc = tf.keras.metrics.BinaryAccuracy(name="train_domain_acc")

    @tf.function
    def _train_step(self, label, ts, y_true, step):
        self.lambda_var.assign(schedule_lambda(step, self.max_steps, self.max_lambda))

        with tf.GradientTape() as tape:
            y_pred, h = self.model((label, ts), lambd=self.lambda_var, training=True, return_latent=True)

            pred_loss = self.mae(y_true, y_pred)

            domain_logit = self.model.classify_domain(h, self.lambda_var)
            label_reshaped = tf.reshape(label, (-1, 1))
            dom_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label_reshaped, domain_logit))

            loss = pred_loss + dom_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_pred_loss.update_state(pred_loss)
        self.train_domain_loss.update_state(dom_loss)
        self.train_domain_acc.update_state(label_reshaped, domain_logit)

    def fit(self, train_ds, val_ds, epochs: int = 1):
        global_step = 0
        for epoch in range(int(epochs)):
            for m in [self.train_loss, self.train_pred_loss, self.train_domain_loss, self.train_domain_acc]:
                if hasattr(m, 'reset_state'):
                    m.reset_state()
                else:
                    m.reset_states()

            for label, ts, y_true in train_ds:
                self._train_step(label, ts, y_true, tf.cast(global_step, tf.float32))
                global_step += 1

            val_loss, val_pred_loss, val_dom_loss, val_dom_acc = self.evaluate(val_ds)

            print(
                f"Epoch {epoch+1:03d} | "
                f"λ={self.lambda_var.numpy():.3f} | "
                f"Train L={self.train_loss.result():.4f} (pred {self.train_pred_loss.result():.4f}, dom {self.train_domain_loss.result():.4f}, acc {self.train_domain_acc.result():.3f}) | "
                f"Val L={val_loss:.4f} (pred {val_pred_loss:.4f}, dom {val_dom_loss:.4f}, acc {val_dom_acc:.3f})"
            )

    @tf.function
    def _val_step(self, label, ts, y_true):
        y_pred, h = self.model((label, ts), lambd=self.lambda_var, training=False, return_latent=True)
        pred_loss = self.mae(y_true, y_pred)
        domain_logit = self.model.classify_domain(h, self.lambda_var)
        dom_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.reshape(label, (-1, 1)), domain_logit))
        return pred_loss + dom_loss, pred_loss, dom_loss, domain_logit

    def evaluate(self, val_ds):
        val_loss = tf.keras.metrics.Mean()
        val_pred_loss = tf.keras.metrics.Mean()
        val_dom_loss = tf.keras.metrics.Mean()
        val_dom_acc = tf.keras.metrics.BinaryAccuracy()

        for label, ts, y_true in val_ds:
            loss, pl, dl, dom_logit = self._val_step(label, ts, y_true)
            val_loss.update_state(loss)
            val_pred_loss.update_state(pl)
            val_dom_loss.update_state(dl)
            val_dom_acc.update_state(tf.reshape(label, (-1, 1)), dom_logit)

        return (
            float(val_loss.result().numpy()),
            float(val_pred_loss.result().numpy()),
            float(val_dom_loss.result().numpy()),
            float(val_dom_acc.result().numpy()),
        )
