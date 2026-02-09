"""Counterfactual evaluation metrics used in the notebooks.

The functions in this module are intentionally dataset-agnostic and accept
series shaped as ``(N, T)`` or ``(N, T, 1)``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional
import numpy as np

try:
    from sklearn.metrics import accuracy_score
except Exception:  # pragma: no cover
    accuracy_score = None


def to_2d_series(y: np.ndarray) -> np.ndarray:
    """(N,T,1)->(N,T); (N,T)->(N,T)"""
    y = np.asarray(y)
    if y.ndim == 3 and y.shape[-1] == 1:
        return y[..., 0]
    if y.ndim == 2:
        return y
    raise ValueError(f"Expected (N,T) or (N,T,1); got {y.shape}")


def to_3d_series(y: np.ndarray) -> np.ndarray:
    """(N,T)->(N,T,1); (N,T,1)->(N,T,1)"""
    y = np.asarray(y)
    if y.ndim == 2:
        return y[..., None]
    if y.ndim == 3 and y.shape[-1] == 1:
        return y
    raise ValueError(f"Expected (N,T) or (N,T,1); got {y.shape}")


def to_1d_labels(l: np.ndarray) -> np.ndarray:
    """(N,1)->(N,); (N,)->(N,)"""
    l = np.asarray(l)
    if l.ndim == 2 and l.shape[1] == 1:
        return l[:, 0]
    if l.ndim == 1:
        return l
    raise ValueError(f"Expected (N,) or (N,1); got {l.shape}")


def counterfactual_mae_mbe(y_true_cf: np.ndarray, y_pred_cf: np.ndarray) -> Dict[str, float]:
    """
    Ground-truth counterfactual metrics
    - MAE: mean absolute error over all steps
    - MBE: mean bias error (mean signed error) over all steps
    """
    yt = to_2d_series(y_true_cf)
    yp = to_2d_series(y_pred_cf)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    return {"cf_mae": mae, "cf_mbe": mbe}


def _predict_scores(predictor, series: np.ndarray) -> np.ndarray:
    """
    Call a predictor that may expect (N,T) or (N,T,1).
    Returns a flat (N,) score array.
    """
    s2 = to_2d_series(series)
    s3 = to_3d_series(series)

    # Try to infer expected rank from Keras-like models
    expected_rank = None
    inp_shape = getattr(predictor, "input_shape", None)
    if isinstance(inp_shape, tuple):
        expected_rank = len(inp_shape)
    elif isinstance(inp_shape, list) and inp_shape and isinstance(inp_shape[0], tuple):
        expected_rank = len(inp_shape[0])

    candidates = []
    if expected_rank == 3:
        candidates = [s3, s2]
    elif expected_rank == 2:
        candidates = [s2, s3]
    else:
        # unknown: try "as-is" first, then the two canonical forms
        candidates = [np.asarray(series), s3, s2]

    last_err = None
    for cand in candidates:
        try:
            out = predictor(cand)
            out = np.asarray(out)
            return out.reshape(-1)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not call predictor on series. Last error: {last_err}")


def axiomatic_metrics(
    model,
    predictor,
    label_real: np.ndarray,
    label_cf: np.ndarray,
    x: np.ndarray,
    y_factual: np.ndarray,
    *,
    composition_arg: int = 1,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Axiomatic metrics used in the experiments:
      - Composition: mean_t |y_factual - composition(...)|
      - Reversibility: mean_t |y_factual - reversibility(...)|
      - Effectiveness: accuracy of event predictor on generated counterfactuals

    Assumes `model` exposes:
      - composition(label_real, x, y, composition_arg)
      - reversibility(label_real, label_cf, x, y, composition_arg)
      - cf_generation(label_real=..., label_cf=..., x=..., y=...)
    """
    y_f = to_2d_series(y_factual)

    reconstruction = to_2d_series(model.composition(label_real, x, y_factual, composition_arg))
    reversibility = to_2d_series(model.reversibility(label_real, label_cf, x, y_factual, composition_arg))

    comp = float(np.mean(np.mean(np.abs(y_f - reconstruction), axis=1)))
    rev = float(np.mean(np.mean(np.abs(y_f - reversibility), axis=1)))

    cf_est = np.asarray(model.cf_generation(label_real=label_real, label_cf=label_cf, x=x, y=y_factual))
    # Keep cf_est as-is for predictor; _predict_scores will adapt rank if needed.
    scores = _predict_scores(predictor, cf_est)

    y_lab = to_1d_labels(label_cf)
    if accuracy_score is None:
        acc = float(np.mean((scores >= threshold) == (y_lab >= 0.5)))
    else:
        acc = float(accuracy_score((y_lab >= 0.5).astype(int), (scores >= threshold).astype(int)))

    return {"composition": comp, "reversibility": rev, "effectiveness": acc}


def added_variations_relative(
    cf_from_y: Callable[[np.ndarray], np.ndarray],
    *,
    y_factual: np.ndarray,
    seq_length: int,
    ini_start: int,
    num_windows: int,
    window_len: int,
    values: Optional[np.ndarray] = None,
    skip_zero: bool = True,
) -> Dict[str, float]:
    """
    Added variations metrics by perturbing a contiguous block of the factual series and
    measuring how much the generated CF changes:

      total_rel      = E[ sum(altered_cf - cf0) ] / ((fin-ini) * val)
      altered_rel    = E[ sum_{ini:fin}(altered_cf - cf0) ] / ((fin-ini) * val)
      unaltered_rel  = E[ sum_{outside}(altered_cf - cf0) ] / ((fin-ini) * val)
    """
    if values is None:
        values = np.arange(-1.0, 1.0 + 1e-9, 0.1)

    y = np.asarray(y_factual)
    y2d = to_2d_series(y)
    n, T = y2d.shape
    if T != seq_length:
        raise ValueError(f"seq_length={seq_length} but y_factual has T={T}")

    input_is_3d = (y.ndim == 3)

    cf0 = to_2d_series(np.asarray(cf_from_y(y)))

    total_rel = []
    altered_rel = []
    unaltered_rel = []

    for i in range(num_windows):
        ini = ini_start + i
        fin = ini + window_len
        if fin > seq_length:
            break

        for val in values:
            if skip_zero and abs(val) < 1e-12:
                continue

            alteration = np.zeros((n, seq_length), dtype=float)
            alteration[:, ini:fin] = val
            altered_actuals_2d = y2d + alteration
            altered_input = altered_actuals_2d[..., None] if input_is_3d else altered_actuals_2d

            altered_cf = to_2d_series(np.asarray(cf_from_y(altered_input)))

            dif_ideal = (fin - ini) * val
            if abs(dif_ideal) < 1e-12:
                continue

            dif_total = np.mean(np.sum(altered_cf - cf0, axis=1)) / dif_ideal
            dif_alt = np.mean(np.sum(altered_cf[:, ini:fin] - cf0[:, ini:fin], axis=1)) / dif_ideal
            dif_unalt = (
                np.mean(
                    np.sum(altered_cf[:, :ini] - cf0[:, :ini], axis=1)
                    + np.sum(altered_cf[:, fin:] - cf0[:, fin:], axis=1)
                )
                / dif_ideal
            )

            total_rel.append(float(dif_total))
            altered_rel.append(float(dif_alt))
            unaltered_rel.append(float(dif_unalt))

    return {
        "total_rel": float(np.mean(total_rel)) if total_rel else float("nan"),
        "altered_steps_rel": float(np.mean(altered_rel)) if altered_rel else float("nan"),
        "unaltered_steps_rel": float(np.mean(unaltered_rel)) if unaltered_rel else float("nan"),
    }
