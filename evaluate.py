"""Evaluation helpers: metrics, confusion matrix, learning curves, mistakes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PathLike = Union[str, Path]


def _finish(fig: plt.Figure, path: Optional[Path], show: bool) -> None:
    """Save the figure if a path is given, optionally show it, then free memory."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_history(history: Mapping[str, Sequence[float]], out_dir: Optional[PathLike] = None,
                 show: bool = False) -> None:
    """Training vs validation accuracy and loss (two separate figures)."""
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history[metric], label="Training")
        ax.plot(history[f"val_{metric}"], label="Validation")
        ax.set(title=f"Training vs validation {metric}", xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)
        _finish(fig, Path(out_dir) / f"{metric}_curve.png" if out_dir else None, show)


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str],
                          out_dir: Optional[PathLike] = None, show: bool = False) -> np.ndarray:
    """Heatmap of true label (rows) vs predicted label (columns)."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    size = max(6.0, n * 0.3)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(cm, annot=n <= 20, fmt="d", cmap="Blues", cbar=n > 20,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title="Confusion matrix", xlabel="Predicted", ylabel="True")
    _finish(fig, Path(out_dir) / "confusion_matrix.png" if out_dir else None, show)
    return cm


def top_confusions(y_true, y_pred, class_names: Sequence[str], top_n: int = 10) -> pd.DataFrame:
    """The most frequent mistakes, e.g. 'true 4, predicted 9: 12 times'."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)  # ignore correct predictions
    order = np.argsort(cm, axis=None)[::-1][:top_n]
    rows = [(class_names[i], class_names[j], int(cm[i, j]))
            for i, j in zip(*np.unravel_index(order, cm.shape)) if cm[i, j] > 0]
    return pd.DataFrame(rows, columns=["true", "predicted", "count"])


def plot_misclassified(x, y_true, y_pred, class_names: Sequence[str], n: int = 16,
                       out_dir: Optional[PathLike] = None, show: bool = False) -> None:
    """Show the first ``n`` test images the model got wrong."""
    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:n]
    if len(wrong) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(np.squeeze(x[idx]), cmap="gray")
        ax.set_title(f"true {class_names[y_true[idx]]}\npred {class_names[y_pred[idx]]}", fontsize=8)
    fig.suptitle("Misclassified test images")
    _finish(fig, Path(out_dir) / "misclassified_examples.png" if out_dir else None, show)


def evaluate_predictions(y_true, y_pred, class_names: Sequence[str],
                         out_dir: Optional[PathLike] = None, show: bool = False) -> Dict[str, float]:
    """Print/save the classification report and return the headline metrics."""
    labels = list(range(len(class_names)))
    text = classification_report(y_true, y_pred, labels=labels, target_names=list(class_names),
                                 digits=4, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(class_names),
                                   output_dict=True, zero_division=0)
    print(text)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "classification_report.txt").write_text(text)
        top_confusions(y_true, y_pred, class_names).to_csv(Path(out_dir) / "top_confusions.csv", index=False)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir, show)
    macro = report["macro avg"]
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro["precision"]),
            "macro_recall": float(macro["recall"]),
            "macro_f1": float(macro["f1-score"])}
