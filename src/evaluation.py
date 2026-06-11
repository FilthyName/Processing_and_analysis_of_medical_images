"""
evaluation.py — единый модуль оценки (Фаза 2).

Не зависит от torch: принимает уже посчитанные массивы предсказаний
(targets / preds / probs) и список классов, поэтому переиспользуется в
ноутбуках, в dl_experiments.py и в анализе ошибок чекпойнта 7.

Закрывает пункты аудита:
  #5  per-class precision/recall/F1 + support, bootstrap-CI для macro-F1
  #3  отдельная sensitivity (recall) по клинически важному классу (mel)
  #4  матрицы ошибок начисто: нормировка по recall и по precision + путаемые пары

Типовое использование:
    from evaluation import evaluate_predictions
    res = evaluate_predictions(targets, preds, probs, classes,
                               out_dir="plots", clinical_positive="mel")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# Per-class отчёт
# --------------------------------------------------------------------------- #
def per_class_report(targets, preds, classes: Sequence[str]) -> pd.DataFrame:
    """Таблица precision/recall/F1/support по каждому классу + итоговые строки."""
    labels = list(range(len(classes)))
    p, r, f1, sup = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    df = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f1, "support": sup}, index=list(classes)
    ).round(4)

    total_support = int(df["support"].sum())  # фиксируем ДО добавления итоговых строк
    macro = df[["precision", "recall", "f1"]].mean()
    # weighted — взвешенный по support
    w = df["support"] / total_support
    weighted = (df[["precision", "recall", "f1"]].T * w).T.sum()

    df.loc["macro avg"] = [macro["precision"], macro["recall"], macro["f1"], total_support]
    df.loc["weighted avg"] = [weighted["precision"], weighted["recall"], weighted["f1"], total_support]
    df["support"] = df["support"].astype(int)
    return df.round(4)


# --------------------------------------------------------------------------- #
# Клиническая метрика: sensitivity по mel (one-vs-rest)
# --------------------------------------------------------------------------- #
def clinical_metrics(targets, preds, probs, classes: Sequence[str],
                     positive: str = "mel") -> Dict[str, float]:
    """
    Для скрининга меланомы главное — НЕ пропустить mel.
    Считаем sensitivity (recall), specificity, precision, miss-rate и ROC-AUC
    для positive против всех остальных.
    """
    if positive not in classes:
        return {}
    idx = list(classes).index(positive)
    t = np.asarray(targets)
    p = np.asarray(preds)
    y_true = (t == idx).astype(int)
    y_pred = (p == idx).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")          # recall mel
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    miss = fn / (tp + fn) if (tp + fn) else float("nan")          # доля пропущенных mel

    out = {
        f"{positive}_sensitivity": round(sens, 4),
        f"{positive}_specificity": round(spec, 4),
        f"{positive}_precision": round(prec, 4),
        f"{positive}_miss_rate": round(miss, 4),
        f"{positive}_support": int(tp + fn),
    }
    if probs is not None:
        try:
            out[f"{positive}_roc_auc"] = round(
                float(roc_auc_score(y_true, np.asarray(probs)[:, idx])), 4
            )
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Bootstrap доверительные интервалы
# --------------------------------------------------------------------------- #
def bootstrap_ci(targets, preds, metric: str = "macro_f1",
                 n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """
    95%-CI метрики ресэмплингом с возвращением.
    Важно при df=17/vasc=20: показывает, что разница между моделями — не шум.
    """
    t = np.asarray(targets)
    p = np.asarray(preds)
    rng = np.random.default_rng(seed)
    n = len(t)

    def compute(tt, pp):
        if metric == "macro_f1":
            return f1_score(tt, pp, average="macro", zero_division=0)
        if metric == "accuracy":
            return accuracy_score(tt, pp)
        if metric == "weighted_f1":
            return f1_score(tt, pp, average="weighted", zero_division=0)
        raise ValueError(metric)

    point = compute(t, p)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = compute(t[idx], p[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": round(float(point), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


# --------------------------------------------------------------------------- #
# Матрицы ошибок и путаемые пары
# --------------------------------------------------------------------------- #
def most_confused_pairs(targets, preds, classes: Sequence[str]) -> pd.DataFrame:
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    rows = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true": classes[i], "pred": classes[j],
                    "count": int(cm[i, j]), "share_of_true": round(float(cm_norm[i, j]), 4),
                })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _plot_cm(cm, classes, title, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def confusion_matrices(targets, preds, classes: Sequence[str], out_dir: Path):
    """Две матрицы: нормировка по true (recall) и по pred (precision)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))).astype(float)
    cm_recall = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_prec = cm / np.clip(cm.sum(axis=0, keepdims=True), 1, None)
    p1 = _plot_cm(cm_recall, classes, "Confusion matrix (normalized by TRUE / recall)",
                  out_dir / "cm_recall.png")
    p2 = _plot_cm(cm_prec, classes, "Confusion matrix (normalized by PRED / precision)",
                  out_dir / "cm_precision.png")
    return p1, p2


# --------------------------------------------------------------------------- #
# Оркестратор
# --------------------------------------------------------------------------- #
def evaluate_predictions(targets, preds, probs, classes: Sequence[str],
                         out_dir: Optional[Path] = None, clinical_positive: str = "mel",
                         n_boot: int = 1000, seed: int = 42, verbose: bool = True) -> dict:
    """Полная оценка: per-class + клиника + CI + матрицы + путаемые пары."""
    report = per_class_report(targets, preds, classes)
    clinical = clinical_metrics(targets, preds, probs, classes, positive=clinical_positive)
    macro_f1 = bootstrap_ci(targets, preds, "macro_f1", n_boot=n_boot, seed=seed)
    acc = bootstrap_ci(targets, preds, "accuracy", n_boot=n_boot, seed=seed)
    confused = most_confused_pairs(targets, preds, classes)

    cm_paths = None
    if out_dir is not None:
        cm_paths = confusion_matrices(targets, preds, classes, out_dir)
        report.to_csv(Path(out_dir) / "per_class_report.csv")
        confused.to_csv(Path(out_dir) / "confused_pairs.csv", index=False)

    if verbose:
        print("=== PER-CLASS ===")
        print(report.to_string())
        print(f"\nmacro-F1 = {macro_f1['point']}  "
              f"(95% CI {macro_f1['ci_low']}–{macro_f1['ci_high']})")
        print(f"accuracy = {acc['point']}  (95% CI {acc['ci_low']}–{acc['ci_high']})")
        if clinical:
            print(f"\n=== КЛИНИКА ({clinical_positive}) ===")
            for k, v in clinical.items():
                print(f"  {k}: {v}")
        print("\n=== ТОП ПУТАЕМЫХ ПАР (true -> pred) ===")
        print(confused.head(8).to_string(index=False))

    return {
        "per_class": report,
        "clinical": clinical,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confused_pairs": confused,
        "cm_paths": cm_paths,
    }
