"""
threshold_tuning.py — пост-обработка для повышения чувствительности к меланоме.

Идея: у обученной модели ROC-AUC по mel высокий (≈0.84), но sensitivity низкая,
потому что argmax при дисбалансе редко выбирает mel. Не переобучая модель,
вводим правило принятия решения:

    если P(mel) >= t  ->  предсказываем mel
    иначе             ->  argmax по всем классам

Порог t подбирается на val под целевую sensitivity (recall) по mel,
затем применяется к test. Это стандартный приём для скрининга: сознательно
жертвуем precision (больше ложных тревог) ради recall (не пропустить рак).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def predict_with_mel_threshold(probs, mel_idx: int, t: float):
    """argmax, но если вероятность mel >= t — форсим mel."""
    preds = np.asarray(probs).argmax(axis=1)
    preds[np.asarray(probs)[:, mel_idx] >= t] = mel_idx
    return preds


def _mel_sens_prec(targets, preds, mel_idx):
    yt = (np.asarray(targets) == mel_idx)
    yp = (np.asarray(preds) == mel_idx)
    return (recall_score(yt, yp, zero_division=0), precision_score(yt, yp, zero_division=0))


def sweep(targets, probs, classes: Sequence[str], positive="mel",
          grid=None) -> pd.DataFrame:
    """Таблица: порог -> sensitivity/precision по mel + macro-F1 (для выбора по val)."""
    mel = list(classes).index(positive)
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 19), 3)
    rows = []
    for t in grid:
        preds = predict_with_mel_threshold(probs, mel, t)
        sens, prec = _mel_sens_prec(targets, preds, mel)
        rows.append({
            "threshold": float(t),
            f"{positive}_sensitivity": round(sens, 4),
            f"{positive}_precision": round(prec, 4),
            "macro_f1": round(f1_score(targets, preds, average="macro", zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def choose_threshold(val_targets, val_probs, classes: Sequence[str],
                     positive="mel", target_sensitivity=0.80) -> float:
    """
    Максимальный порог, при котором sensitivity по mel на VAL >= цели.
    Берём максимальный (а не любой) — чтобы при заданной чувствительности
    сохранить как можно более высокую precision (меньше ложных тревог).
    """
    mel = list(classes).index(positive)
    grid = np.round(np.linspace(0.02, 0.95, 94), 3)
    ok = []
    for t in grid:
        preds = predict_with_mel_threshold(val_probs, mel, t)
        sens, _ = _mel_sens_prec(val_targets, preds, mel)
        if sens >= target_sensitivity:
            ok.append(t)
    return float(max(ok)) if ok else float(grid[0])


def compare(test_targets, test_probs, classes: Sequence[str], threshold: float,
            positive="mel") -> pd.DataFrame:
    """Метрики до (argmax) и после (с порогом) на test."""
    mel = list(classes).index(positive)
    base = np.asarray(test_probs).argmax(axis=1)
    tuned = predict_with_mel_threshold(test_probs, mel, threshold)

    def row(name, preds):
        sens, prec = _mel_sens_prec(test_targets, preds, mel)
        return {
            "вариант": name,
            f"{positive}_sensitivity": round(sens, 4),
            f"{positive}_miss_rate": round(1 - sens, 4),
            f"{positive}_precision": round(prec, 4),
            "macro_f1": round(f1_score(test_targets, preds, average="macro", zero_division=0), 4),
            "accuracy": round((np.asarray(test_targets) == preds).mean(), 4),
        }

    return pd.DataFrame([row("argmax (было)", base),
                         row(f"порог mel={threshold:.2f}", tuned)])


def plot_tradeoff(targets, probs, classes, out_path, positive="mel"):
    """Кривая sensitivity/precision по mel в зависимости от порога."""
    import matplotlib.pyplot as plt

    df = sweep(targets, probs, classes, positive=positive,
               grid=np.round(np.linspace(0.02, 0.95, 40), 3))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold"], df[f"{positive}_sensitivity"], label="sensitivity (recall)")
    ax.plot(df["threshold"], df[f"{positive}_precision"], label="precision")
    ax.plot(df["threshold"], df["macro_f1"], label="macro-F1", linestyle="--")
    ax.set_xlabel(f"порог P({positive})"); ax.set_ylabel("метрика")
    ax.set_title(f"Trade-off по {positive}"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path
