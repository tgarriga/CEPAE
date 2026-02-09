from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_rossmann_datasets(
    csv_path: str,
    lookback: int = 28,
    horizon: int = 21,
    seed: int = 42,
    e_factual: int = 1,
    event_scale=(1.1, 1.2, 1.3),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create semi-synthetic Rossmann windows and split into train/eval/test sets.

    We create two outcome versions for each window:
      - y0: baseline (no-event)
      - y1: event version obtained by applying a multiplicative intervention

    The *factual* and *counterfactual* views are selected purely by (e_factual, e_counterfactual).
    """
    e_counterfactual = 1 - e_factual

    # --- Load + parse
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Build rolling windows
    x_series, y_series = [], []
    anchor_dates = ("2013-03-04", "2014-03-03", "2015-03-02")
    anchor_dates = pd.to_datetime(list(anchor_dates))

    for store in df["Store"].unique():
        df_store = df[df["Store"] == store].sort_values("Date")
        for date in anchor_dates:
            initial = date - pd.Timedelta(days=lookback)
            final = date + pd.Timedelta(days=horizon)

            x = df_store[(df_store["Date"] >= initial) & (df_store["Date"] < date)]["Sales"].to_numpy()
            y = df_store[(df_store["Date"] >= date)    & (df_store["Date"] < final)]["Sales"].to_numpy()

            # Keep only complete windows
            if len(x) != lookback or len(y) != horizon:
                continue

            # Reverse to match the experimental setup used in the paper
            x_series.append(x[::-1])
            y_series.append(y[::-1])

    x_series = np.asarray(x_series, dtype=np.float32)
    y_series = np.asarray(y_series, dtype=np.float32)

    # --- Normalize by the mean of the lookback window (per-series)
    mean = x_series.mean(axis=1, keepdims=True)
    keep = (mean.squeeze() != 0.0)
    x_series = x_series[keep] / mean[keep]
    y_series = y_series[keep] / mean[keep]

    # --- Split train/eval/test
    rs = seed if seed is not None else None
    train_x, eval_test_x, train_y, eval_test_y = train_test_split(x_series, y_series, test_size=0.2, random_state=rs)
    eval_x, test_x, eval_y, test_y = train_test_split(eval_test_x, eval_test_y, test_size=0.5, random_state=rs)

    # Split train into two domains: no-event (0) and event (1)
    train_x_0, train_x_1, train_y_0, train_y_1 = train_test_split(train_x, train_y, test_size=0.5, random_state=rs)

    def apply_event(y_2d: np.ndarray) -> np.ndarray:
        """Apply the semi-synthetic event intervention."""
        y_out = y_2d.copy()
        a0, a1, a2 = event_scale
        y_out[:, 0] *= a0
        y_out[:, 1] *= a1
        y_out[:, 2:] *= a2
        return y_out

    # Event version for the event domain (train) + for eval/test (paired versions)
    train_y_1_event = apply_event(train_y_1)
    eval_y_event    = apply_event(eval_y)
    test_y_event    = apply_event(test_y)

    # --- Pack train
    x_train = np.concatenate([train_x_0, train_x_1]).reshape(-1, lookback, 1)
    y_train = np.concatenate([train_y_0, train_y_1_event]).reshape(-1, horizon, 1)
    train_labels = np.concatenate(
        [np.zeros((len(train_x_0), 1), dtype=np.float32),
         np.ones((len(train_x_1), 1), dtype=np.float32)],
        axis=0
    )

    # --- Build eval/test paired views (y0=no-event, y1=event)
    x_eval = eval_x.reshape(-1, lookback, 1)
    y0_eval = eval_y.reshape(-1, horizon, 1)
    y1_eval = eval_y_event.reshape(-1, horizon, 1)

    x_test = test_x.reshape(-1, lookback, 1)
    y0_test = test_y.reshape(-1, horizon, 1)
    y1_test = test_y_event.reshape(-1, horizon, 1)

    def select_views(y0, y1, e_f, e_cf):
        y_f  = y1 if int(e_f)  == 1 else y0
        y_cf = y1 if int(e_cf) == 1 else y0
        lab_f  = np.full((len(y0), 1), float(e_f), dtype=np.float32)
        lab_cf = np.full((len(y0), 1), float(e_cf), dtype=np.float32)
        return y_f, y_cf, lab_f, lab_cf

    y_f_eval, y_cf_eval, lab_f_eval, lab_cf_eval = select_views(y0_eval, y1_eval, e_factual, e_counterfactual)
    y_f_test, y_cf_test, lab_f_test, lab_cf_test = select_views(y0_test, y1_test, e_factual, e_counterfactual)

    return {
        "train": {"x": x_train, "y": y_train, "labels": train_labels},
        "eval":  {"x": x_eval, "y0": y0_eval, "y1": y1_eval,
                  "y_factual": y_f_eval, "y_counterfactual": y_cf_eval,
                  "label_factual": lab_f_eval, "label_counterfactual": lab_cf_eval},
        "test":  {"x": x_test, "y0": y0_test, "y1": y1_test,
                  "y_factual": y_f_test, "y_counterfactual": y_cf_test,
                  "label_factual": lab_f_test, "label_counterfactual": lab_cf_test},
    }
