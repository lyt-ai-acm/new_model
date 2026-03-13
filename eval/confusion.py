"""
Confusion matrix visualization for three-class sentiment analysis.
"""
from typing import List, Optional
import numpy as np


LABELS = ["neg", "neu", "pos"]
LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}


def compute_confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    """Returns 3x3 confusion matrix (rows=true, cols=pred)."""
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[LABEL2ID[t]][LABEL2ID[p]] += 1
    return cm


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot and optionally save the confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, printing text confusion matrix instead.")
        cm = compute_confusion_matrix(y_true, y_pred)
        print(f"\n{title}")
        print("       " + "  ".join(LABELS))
        for i, label in enumerate(LABELS):
            print(f"{label:6s} " + "  ".join(f"{cm[i][j]:4d}" for j in range(len(LABELS))))
        return
    
    cm = compute_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()
