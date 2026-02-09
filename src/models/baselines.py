from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, LSTM


class ForecastModel(tf.keras.Model):
    """Simple forecasting baseline.

    Inputs:
        (event, history) where ``event`` has shape (batch, 1) and ``history`` has
        shape (batch, lookback, 1).

    Output:
        A vector of length ``pred_steps`` with the predicted future values.
    """

    def __init__(self, pred_steps: int):
        super().__init__()
        self._pred_steps = int(pred_steps)

        self.lstm1 = LSTM(32, return_sequences=True)
        self.lstm2 = LSTM(32, return_sequences=False)
        self.concat = Concatenate()
        self.dense1 = Dense(32, activation="relu")
        self.dense2 = Dense(self._pred_steps)

    def call(self, inputs, training: bool = False):
        event, history = inputs
        x = self.lstm1(history, training=training)
        x = self.lstm2(x, training=training)
        x = self.concat([event, x])
        x = self.dense1(x, training=training)
        return self.dense2(x, training=training)


class EventPredictor(tf.keras.Model):
    """Binary event predictor from an outcome sequence.

    Used to compute the *effectiveness* metric by evaluating whether generated
    counterfactuals are classified as the intended event state.
    """

    def __init__(self):
        super().__init__()
        self.lstm1 = LSTM(32, return_sequences=True)
        self.lstm2 = LSTM(32, return_sequences=False)
        self.dense1 = Dense(32, activation="relu")
        self.dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs, training: bool = False):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.dense1(x, training=training)
        return self.dense2(x, training=training)
